"""Independent session verification (Milestone 7, Section 26).
`verify_paper_session` NEVER trusts the persisted `PaperSessionManifest`/
final report at face value -- it independently recomputes spec identity,
re-verifies the FULL eligibility chain (Milestone 6 re-recomputation
included), re-checks manifest transition legality against the ledger's
own `SESSION_TRANSITION` entries, verifies ledger hash-chain integrity,
and re-runs `reconciliation.reconcile_session` (itself a from-ledger
recomputation, never trusting persisted totals).

Reuses `ml.models.ValidationReport`/`ValidationIssue`/`ValidationSeverity`
directly -- the SAME report shape `robustness.verification.verify_robustness`
returns one layer down, rather than inventing a second one.

INDEPENDENCE CLASSIFICATION (Section 26's own explicit instruction:
"classify verification honestly") -- `verify_paper_session` is a MIX,
stated plainly rather than claimed uniformly "independent":

  - Spec identity + ledger chain integrity + manifest transition
    legality: STRUCTURALLY INDEPENDENT (pure recomputation from hashes/
    enums, no financial logic, cannot be fooled by a wrong NUMBER, only
    by a wrong STRUCTURE).
  - Eligibility chain: SOURCE-RECONSTRUCTING -- `eligibility.
    verify_paper_trading_eligibility` calls straight through to
    `robustness.verification.verify_robustness`/`backtesting.
    verification.verify_backtest`, which themselves recompute every
    statistic from RAW evidence, not merely re-reading a persisted total.
  - Reconciliation (position/cash/costs/equity): ALGORITHMICALLY
    INDEPENDENT -- `reconciliation.reconcile_session` recomputes account
    state using the SAME accounting formulas as the forward run, from the
    ledger's own persisted `Fill`/`FINANCING_APPLIED` entries. This is
    NOT source-reconstructing for the STRATEGY's decisions themselves:
    it takes the ledger's persisted `StrategyDecision`/`Fill` records as
    given and checks the ACCOUNTING built from them balances, but it does
    NOT re-invoke the original `StrategyRuntime` against raw market data
    a second time to confirm the SAME decisions would be reached again --
    that specific check (true decision-level re-execution) is the real
    acceptance workflow's own "run twice, compare digests" property
    (Section 33), not this function's.

Net honest classification for the WHOLE report: **PARTIALLY INDEPENDENT**
-- strong (source-reconstructing) for eligibility, strong (algorithmically
independent, from-ledger) for accounting, but NOT a full re-execution of
strategy decision-making from raw market data. `PaperSessionVerification
Report.independence_classification` states this explicitly rather than
letting a bare `is_ready=True` imply more than was actually checked."""

from __future__ import annotations

from quant_platform.core.exceptions import PaperTradingArtifactError, PaperTradingVerificationError
from quant_platform.ml.models import ValidationIssue, ValidationReport, ValidationSeverity
from quant_platform.ml.persistence import format_utc_timestamp, utc_now
from quant_platform.paper_trading.eligibility import (
    EligibilityVerificationEnvironment,
    verify_paper_trading_eligibility,
)
from quant_platform.paper_trading.manifests import PaperSessionManifest
from quant_platform.paper_trading.models import (
    TERMINAL_PAPER_SESSION_STAGES,
    LedgerEntryKind,
    PaperSessionStage,
    SessionMode,
    is_legal_paper_session_transition,
)
from quant_platform.paper_trading.persistence import LedgerEntry, verify_ledger_chain_integrity
from quant_platform.paper_trading.reconciliation import reconcile_session
from quant_platform.paper_trading.specs import PaperTradingSpec, compute_paper_session_spec_id

VERIFICATION_REPORT_SCHEMA_VERSION = 1

INDEPENDENCE_CLASSIFICATION = (
    "PARTIALLY INDEPENDENT: spec identity/ledger chain integrity/manifest transition legality are STRUCTURALLY "
    "INDEPENDENT; the eligibility chain is SOURCE-RECONSTRUCTING (recomputes from raw Milestone 6 evidence); "
    "reconciliation (position/cash/costs/equity) is ALGORITHMICALLY INDEPENDENT (recomputes from the ledger's own "
    "persisted fills using the same accounting formulas, but does not re-invoke the original StrategyRuntime against "
    "raw market data a second time -- that decision-level re-execution is the real acceptance workflow's own "
    "run-twice-compare-digests property, not this function's)."
)


def _issue(severity: ValidationSeverity, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message)


def _verify_manifest_transitions(ledger: list[LedgerEntry], *, manifest: PaperSessionManifest) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    transition_entries = [e for e in ledger if e.kind is LedgerEntryKind.SESSION_TRANSITION]
    current = PaperSessionStage.CREATED
    for entry in transition_entries:
        from_stage = PaperSessionStage(entry.payload["from_stage"])
        to_stage = PaperSessionStage(entry.payload["to_stage"])
        if from_stage is not current:
            issues.append(_issue(ValidationSeverity.CRITICAL, "session_transition_out_of_order", f"ledger SESSION_TRANSITION entry {entry.entry_id!r} expects from_stage={from_stage.value!r} but replay is currently at {current.value!r}"))
            return issues
        if not is_legal_paper_session_transition(current, to_stage):
            issues.append(_issue(ValidationSeverity.CRITICAL, "illegal_session_transition", f"ledger records illegal transition {current.value!r} -> {to_stage.value!r}"))
            return issues
        current = to_stage
    if current is not manifest.stage:
        issues.append(_issue(ValidationSeverity.ERROR, "manifest_stage_mismatch", f"manifest.stage={manifest.stage.value!r} does not match the stage reconstructed from the ledger's own SESSION_TRANSITION entries ({current.value!r})"))
    return issues


def _verify_market_event_ordering(ledger: list[LedgerEntry]) -> list[ValidationIssue]:
    """Release-audit finding, fixed (Section 7): `replay.validate_replay_
    sequence` enforces strictly-increasing sequence numbers, unique
    event ids, and chronological order on the ORIGINAL source file,
    ONCE, before a session even begins -- but nothing ever re-checked
    that the PERSISTED ledger's own `MARKET_EVENT_ACCEPTED` entries
    still have this shape afterward. Neither `reconcile_session` nor any
    other step in this module reads a `MARKET_EVENT_ACCEPTED` entry's
    own payload at all, so a ledger with two events swapped (or an
    event's identity changed), with fully valid RE-CHAINED hashes,
    previously passed every existing check -- exactly the audit's own
    named blocker class: "hash-valid semantic ledger corruption
    accepted". This does not change any FINANCIAL total (fills/P&L
    already reflect whatever actually happened), but it corrupts the
    ledger's own evidentiary record of which market data produced which
    decision, which this independent verification must catch."""
    issues: list[ValidationIssue] = []
    seen_event_ids: set[str] = set()
    seen_sequences: set[int] = set()
    previous_sequence: int | None = None
    previous_event_time = None
    for entry in ledger:
        if entry.kind is not LedgerEntryKind.MARKET_EVENT_ACCEPTED:
            continue
        event_id = str(entry.payload.get("event_id"))
        sequence = int(str(entry.payload.get("sequence")))
        if event_id in seen_event_ids:
            issues.append(_issue(ValidationSeverity.CRITICAL, "duplicate_market_event_identity", f"ledger contains a duplicate MARKET_EVENT_ACCEPTED event_id {event_id!r} (entry_id={entry.entry_id!r})"))
        seen_event_ids.add(event_id)
        if sequence in seen_sequences:
            issues.append(_issue(ValidationSeverity.CRITICAL, "duplicate_market_event_sequence", f"ledger contains a duplicate MARKET_EVENT_ACCEPTED sequence {sequence} (entry_id={entry.entry_id!r})"))
        seen_sequences.add(sequence)
        if previous_sequence is not None and sequence <= previous_sequence:
            issues.append(_issue(ValidationSeverity.CRITICAL, "market_event_sequence_not_increasing", f"ledger MARKET_EVENT_ACCEPTED sequence is not strictly increasing: {previous_sequence} -> {sequence} (entry_id={entry.entry_id!r})"))
        previous_sequence = sequence
        if previous_event_time is not None and entry.event_time < previous_event_time:
            issues.append(_issue(ValidationSeverity.CRITICAL, "market_event_not_chronological", f"ledger MARKET_EVENT_ACCEPTED is not in chronological order: {previous_event_time} -> {entry.event_time} (entry_id={entry.entry_id!r})"))
        previous_event_time = entry.event_time
    return issues


def _verify_ledger_entries_belong_to_session(ledger: list[LedgerEntry], *, paper_session_id: str) -> list[ValidationIssue]:
    """Release-audit finding, fixed (Section 7): nothing previously
    checked that every `LedgerEntry.session_id` in `ledger` actually
    equals the session being verified -- `verify_paper_session` takes
    `ledger` as a plain `list[LedgerEntry]` argument, entirely decoupled
    from how the caller obtained it. Feeding it a DIFFERENT session's
    ledger (by accident, or "use a ledger from another session" as a
    deliberate attack) was undetected: every other check here recomputes
    FROM the ledger's own content, never cross-checking which session
    that content actually claims to belong to."""
    foreign_entry_ids = sorted({e.entry_id for e in ledger if e.session_id != paper_session_id})
    if not foreign_entry_ids:
        return []
    return [_issue(ValidationSeverity.CRITICAL, "ledger_entry_belongs_to_another_session", f"{len(foreign_entry_ids)} ledger entr{'y' if len(foreign_entry_ids) == 1 else 'ies'} declare a session_id other than {paper_session_id!r}: {foreign_entry_ids[:5]}{'...' if len(foreign_entry_ids) > 5 else ''}")]


_REAL_ACCOUNT_ONLY_KINDS = (
    LedgerEntryKind.ORDER_STATE_EVENT, LedgerEntryKind.FILL, LedgerEntryKind.FINANCING_APPLIED, LedgerEntryKind.RISK_DECISION,
    LedgerEntryKind.HALT_TRIGGERED, LedgerEntryKind.ACCOUNT_SNAPSHOT, LedgerEntryKind.RECONCILIATION_RESULT,
)


def _verify_ledger_matches_session_mode(ledger: list[LedgerEntry], *, session_mode: SessionMode) -> list[ValidationIssue]:
    """Release-audit finding, fixed (Section 5): nothing previously
    checked that a ledger's own entry KINDS are consistent with its
    session's declared `session_mode` -- `run_shadow_session` never
    constructs an `ORDER_STATE_EVENT`/`FILL`/`ACCOUNT_SNAPSHOT`/etc. and
    `run_paper_trading_session` never constructs a `SHADOW_OBSERVATION`,
    but that was only ever true BY CONSTRUCTION of the trusted forward
    code path -- this is the INDEPENDENT verification layer, which must
    not simply assume it. A hand-tampered or wrongly-spliced ledger
    (real fills injected into a shadow session's ledger, or vice versa)
    previously had no structural check catching it: `reconcile_session`
    against a fill-free shadow ledger trivially "reconciles" (zero fills,
    zero discrepancy), so `is_ready=True` could be reached without ever
    inspecting whether the ledger's CONTENT actually matches what its
    own declared mode permits."""
    issues: list[ValidationIssue] = []
    if session_mode is SessionMode.SHADOW_OBSERVATION:
        offending_kinds = sorted({e.kind.value for e in ledger if e.kind in _REAL_ACCOUNT_ONLY_KINDS})
        if offending_kinds:
            issues.append(_issue(ValidationSeverity.CRITICAL, "shadow_session_ledger_contains_real_account_entries", f"session_mode=shadow_observation but the ledger contains real-account-only entry kind(s): {offending_kinds}"))
    else:
        shadow_entry_count = sum(1 for e in ledger if e.kind is LedgerEntryKind.SHADOW_OBSERVATION)
        if shadow_entry_count:
            issues.append(_issue(ValidationSeverity.CRITICAL, "paper_session_ledger_contains_shadow_entries", f"session_mode={session_mode.value!r} but the ledger contains {shadow_entry_count} SHADOW_OBSERVATION entry/entries"))
    return issues


def verify_paper_session(
    spec: PaperTradingSpec, *, manifest: PaperSessionManifest, ledger: list[LedgerEntry], eligibility_environment: EligibilityVerificationEnvironment,
) -> ValidationReport:
    issues: list[ValidationIssue] = []

    # Step: verify spec identity.
    recomputed_id = compute_paper_session_spec_id(spec).paper_session_spec_id
    if recomputed_id != manifest.paper_session_id:
        issues.append(_issue(ValidationSeverity.CRITICAL, "spec_identity_mismatch", f"recomputed paper_session_spec_id={recomputed_id!r} does not match manifest.paper_session_id={manifest.paper_session_id!r}"))

    # Step: verify every ledger entry actually belongs to THIS session.
    issues.extend(_verify_ledger_entries_belong_to_session(ledger, paper_session_id=manifest.paper_session_id))

    # Step: verify the ledger's own entry kinds are consistent with the declared session_mode.
    issues.extend(_verify_ledger_matches_session_mode(ledger, session_mode=spec.session_mode))

    # Step: verify the ledger's own MARKET_EVENT_ACCEPTED entries are still correctly ordered/unique.
    issues.extend(_verify_market_event_ordering(ledger))

    # Step: verify eligibility chain (source-reconstructing).
    try:
        eligibility_report = verify_paper_trading_eligibility(spec, environment=eligibility_environment)
        if not eligibility_report.is_eligible:
            issues.append(_issue(ValidationSeverity.CRITICAL, "eligibility_not_verified", f"eligibility re-verification failed at step {eligibility_report.failed_step!r}: {eligibility_report.failure_reason}"))
    except (PaperTradingArtifactError, ValueError, KeyError, TypeError) as exc:
        issues.append(_issue(ValidationSeverity.CRITICAL, "eligibility_verification_raised", f"eligibility re-verification raised: {exc}"))

    # Step: verify manifest transitions.
    issues.extend(_verify_manifest_transitions(ledger, manifest=manifest))

    # Step: verify ledger integrity.
    try:
        verify_ledger_chain_integrity(ledger)
    except PaperTradingArtifactError as exc:
        issues.append(_issue(ValidationSeverity.CRITICAL, "ledger_chain_broken", str(exc)))

    # Step: replay ledger from initial account state, reconstruct
    # orders/fills/positions/costs/account snapshots, compare to persisted
    # (reconciliation IS this replay-and-compare, algorithmically
    # independent from the ledger's own persisted fills).
    reconciliation_report = reconcile_session(ledger, session_id=manifest.paper_session_id, instrument=spec.instrument, starting_cash=spec.starting_cash)
    for check in reconciliation_report.checks:
        if not check.passed:
            issues.append(_issue(ValidationSeverity.CRITICAL, f"reconciliation_{check.check_identity}_failed", f"{check.check_identity}: expected={check.expected_value!r} observed={check.observed_value!r} (tolerance={check.tolerance!r})"))

    # Step: verify final reconciliation persisted-vs-recomputed (if a
    # RECONCILIATION_RESULT was itself persisted to the ledger).
    persisted_reconciliation_entries = [e for e in ledger if e.kind is LedgerEntryKind.RECONCILIATION_RESULT]
    if persisted_reconciliation_entries:
        persisted_status = persisted_reconciliation_entries[-1].payload
        if bool(persisted_status.get("is_reconciled")) != reconciliation_report.is_reconciled:
            issues.append(_issue(ValidationSeverity.CRITICAL, "reconciliation_status_mismatch", "persisted reconciliation status does not match the independently recomputed one"))

    # Step: verify session actually reached a terminal stage consistent with the manifest.
    if manifest.stage not in TERMINAL_PAPER_SESSION_STAGES:
        issues.append(_issue(ValidationSeverity.WARNING, "session_not_terminal", f"manifest.stage={manifest.stage.value!r} is not a terminal stage -- verification of an in-progress session is necessarily partial"))

    return ValidationReport(schema_version=VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(utc_now()))


def require_paper_session_verified(
    spec: PaperTradingSpec, *, manifest: PaperSessionManifest, ledger: list[LedgerEntry], eligibility_environment: EligibilityVerificationEnvironment,
) -> ValidationReport:
    report = verify_paper_session(spec, manifest=manifest, ledger=ledger, eligibility_environment=eligibility_environment)
    if not report.is_ready:
        codes = ", ".join(sorted({i.code for i in report.criticals} | {i.code for i in report.errors}))
        raise PaperTradingVerificationError(f"Paper session {manifest.paper_session_id!r} failed independent verification: {codes}")
    return report


__all__ = [
    "INDEPENDENCE_CLASSIFICATION",
    "VERIFICATION_REPORT_SCHEMA_VERSION",
    "require_paper_session_verified",
    "verify_paper_session",
]
