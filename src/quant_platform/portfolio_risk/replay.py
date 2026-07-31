"""Deterministic replay comparison for `quant_platform.portfolio_risk`
(Milestone 9, Phase 3). This package has no session RUNNER to replay
wholesale (unlike `execution_gateway.replay`, which re-runs a whole
execution session end to end) -- Phase 3's own "session" is simply
"every ledger entry recorded for one `portfolio_id`", built by
individual, discrete calls to `lifecycle.py`'s transaction functions.
`compute_replay_result`/`assert_replay_deterministic` therefore provide
the COMPARISON primitive tests use to prove that replaying the SAME
sequence of operations into a FRESH, independent
`PortfolioRiskLedgerStore` (a different temp directory, a different
process, a different `PYTHONHASHSEED`) produces byte-identical economic
outcomes -- never that this module itself re-executes anything."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quant_platform.core.exceptions import PortfolioRiskVerificationError
from quant_platform.portfolio_risk.idempotency import build_authorization_payload_index
from quant_platform.portfolio_risk.ledger import PortfolioRiskLedgerStore, compute_risk_ledger_semantic_digest
from quant_platform.portfolio_risk.reconciliation import reconcile_portfolio_risk_session
from quant_platform.portfolio_risk.verification import verify_portfolio_risk_session

__all__ = ["PortfolioRiskReplayResult", "assert_replay_deterministic", "compute_replay_result"]


@dataclass(frozen=True, slots=True)
class PortfolioRiskReplayResult:
    portfolio_id: str
    semantic_digest: str
    authorization_ids: tuple[str, ...]
    critical_issue_count: int
    is_reconciled: bool


def compute_replay_result(*, portfolio_id: str, store: PortfolioRiskLedgerStore, verification_time: datetime) -> PortfolioRiskReplayResult:
    ledger = store.read_events(portfolio_id)
    semantic_digest = compute_risk_ledger_semantic_digest(ledger)
    authorization_ids = tuple(sorted(build_authorization_payload_index(ledger).keys()))
    reconciliation_report = reconcile_portfolio_risk_session(portfolio_id=portfolio_id, ledger=ledger)
    verification_report = verify_portfolio_risk_session(portfolio_id=portfolio_id, store=store, verification_time=verification_time, record=False)
    return PortfolioRiskReplayResult(
        portfolio_id=portfolio_id, semantic_digest=semantic_digest, authorization_ids=authorization_ids,
        critical_issue_count=len(verification_report.criticals), is_reconciled=reconciliation_report.is_reconciled,
    )


def assert_replay_deterministic(a: PortfolioRiskReplayResult, b: PortfolioRiskReplayResult) -> None:
    if a.semantic_digest != b.semantic_digest:
        raise PortfolioRiskVerificationError(f"Replay divergence: semantic digests differ ({a.semantic_digest!r} != {b.semantic_digest!r})")
    if a.authorization_ids != b.authorization_ids:
        raise PortfolioRiskVerificationError(f"Replay divergence: authorization id sets differ ({a.authorization_ids!r} != {b.authorization_ids!r})")
    if a.critical_issue_count != b.critical_issue_count:
        raise PortfolioRiskVerificationError("Replay divergence: verification critical counts differ")
    if a.is_reconciled != b.is_reconciled:
        raise PortfolioRiskVerificationError("Replay divergence: reconciliation outcome differs")
