"""Deterministic replay (Milestone 8, Section 30). `replay_execution_session`
runs `spec` end-to-end against a FRESH, isolated store (its own temporary
`ExecutionSessionManifestStore`/`ExecutionSessionEventStore` and a fresh
`DeterministicDummyBrokerAdapter`) and returns one `ReplayResult` bundling
everything a determinism comparison needs: the final semantic digest,
final order states, fills, broker snapshots, reconciliation, and
verification. Two calls with the SAME immutable inputs (spec, source
paper orders, market events, dummy broker scenario, seed) -- even across
separate processes, different `PYTHONHASHSEED` values, and different
temporary storage paths -- must produce the SAME `ReplayResult.
semantic_digest` and the SAME economic/state outputs; this is the
property `tests/unit/execution_gateway/test_deterministic_replay.py` and
the acceptance workflow's own second-process run both check."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quant_platform.execution_gateway.adapter import ExecutionAdapter
from quant_platform.execution_gateway.dummy_broker import DeterministicDummyBrokerAdapter
from quant_platform.execution_gateway.manifests import ExecutionSessionManifest, ExecutionSessionManifestStore
from quant_platform.execution_gateway.paper_bridge import PaperBridgeEnvironment
from quant_platform.execution_gateway.persistence import ExecutionSessionEventStore
from quant_platform.execution_gateway.portfolio_risk_gate import PortfolioRiskGatewayContext
from quant_platform.execution_gateway.reconciliation import (
    ExecutionReconciliationReport,
    reconcile_execution_session,
)
from quant_platform.execution_gateway.reports import ExecutionSessionReport, generate_execution_session_report
from quant_platform.execution_gateway.runner import RunnerEnvironment, run_execution_session
from quant_platform.execution_gateway.specs import ExecutionGatewaySpec
from quant_platform.execution_gateway.verification import verify_execution_session
from quant_platform.ml.models import ValidationReport
from quant_platform.paper_trading.events import MarketEvent
from quant_platform.paper_trading.orders import OrderRequest


@dataclass(frozen=True, slots=True)
class ReplayResult:
    execution_session_id: str
    manifest: ExecutionSessionManifest
    semantic_digest: str | None
    reconciliation_report: ExecutionReconciliationReport
    verification_report: ValidationReport
    session_report: ExecutionSessionReport


def replay_execution_session(
    spec: ExecutionGatewaySpec, *, storage_root: Path, paper_bridge_environment: PaperBridgeEnvironment, portfolio_risk_context: PortfolioRiskGatewayContext,
    paper_orders: list[OrderRequest], market_events: Iterable[MarketEvent], adapter_id: str, event_time: datetime,
) -> ReplayResult:
    """`portfolio_risk_context` (Milestone 9 Phase 4) is threaded straight
    through to `RunnerEnvironment`, exactly like `paper_bridge_environment`
    already is -- the CALLER is responsible for its `store` pointing at a
    fresh, isolated root when determinism is what's being tested (mirrors
    `manifest_store`/`event_store` below, both freshly constructed from
    THIS call's own `storage_root`); `replay_execution_session` itself
    does not second-guess or rebuild it."""
    manifest_store = ExecutionSessionManifestStore(storage_root)
    event_store = ExecutionSessionEventStore(storage_root)
    environment = RunnerEnvironment(
        manifest_store=manifest_store, event_store=event_store, paper_bridge_environment=paper_bridge_environment, portfolio_risk_context=portfolio_risk_context,
    )
    adapter: ExecutionAdapter = DeterministicDummyBrokerAdapter(adapter_id=adapter_id, scenario=spec.dummy_broker_scenario)

    manifest = run_execution_session(spec, environment=environment, adapter=adapter, paper_orders=paper_orders, market_events=market_events, event_time=event_time)
    ledger = event_store.read_events(manifest.execution_session_id)

    reconciliation_report = reconcile_execution_session(execution_session_id=manifest.execution_session_id, ledger=ledger, adapter=adapter, event_time=event_time, policy=spec.reconciliation_policy)
    verification_report = verify_execution_session(spec, execution_session_id=manifest.execution_session_id, ledger=ledger)
    session_report = generate_execution_session_report(execution_session_id=manifest.execution_session_id, ledger=ledger, reconciliation_report=reconciliation_report)

    return ReplayResult(
        execution_session_id=manifest.execution_session_id, manifest=manifest, semantic_digest=manifest.semantic_digest, reconciliation_report=reconciliation_report,
        verification_report=verification_report, session_report=session_report,
    )


__all__ = ["ReplayResult", "replay_execution_session"]
