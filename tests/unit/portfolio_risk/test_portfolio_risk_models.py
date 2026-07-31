"""Unit tests for `portfolio_risk.models`: the closed vocabularies every
other module in this package imports."""

from __future__ import annotations

from quant_platform.portfolio_risk.models import (
    RiskAuthorizationStatus,
    RiskCheckSeverity,
    RiskDecisionKind,
    is_legal_risk_authorization_status_transition,
    is_terminal_risk_authorization_status,
    most_severe_check_severity,
)


class TestRiskDecisionKindHasNoUnknownValue:
    def test_exactly_three_members(self) -> None:
        assert {k.value for k in RiskDecisionKind} == {"approved", "denied", "halted"}

    def test_no_unknown_member_exists_structurally(self) -> None:
        # There is no way to even reference a non-existent enum member --
        # this test documents that "no UNKNOWN approval state" is a
        # structural property of the enum's own definition, not merely an
        # unused convention.
        assert not hasattr(RiskDecisionKind, "UNKNOWN")
        assert not hasattr(RiskDecisionKind, "PENDING")


class TestRiskCheckSeverityOrdering:
    def test_empty_defaults_to_info(self) -> None:
        assert most_severe_check_severity(()) is RiskCheckSeverity.INFO

    def test_max_wins(self) -> None:
        assert most_severe_check_severity((RiskCheckSeverity.INFO, RiskCheckSeverity.WARNING)) is RiskCheckSeverity.WARNING
        assert most_severe_check_severity((RiskCheckSeverity.WARNING, RiskCheckSeverity.DENY)) is RiskCheckSeverity.DENY
        assert most_severe_check_severity((RiskCheckSeverity.DENY, RiskCheckSeverity.HALT)) is RiskCheckSeverity.HALT

    def test_order_of_arguments_does_not_matter(self) -> None:
        a = most_severe_check_severity((RiskCheckSeverity.HALT, RiskCheckSeverity.INFO, RiskCheckSeverity.DENY))
        b = most_severe_check_severity((RiskCheckSeverity.DENY, RiskCheckSeverity.INFO, RiskCheckSeverity.HALT))
        assert a is b is RiskCheckSeverity.HALT


class TestRiskAuthorizationStatusTransitions:
    def test_issued_can_transition_to_reserved_or_directly_to_a_terminal_status(self) -> None:
        for target in (
            RiskAuthorizationStatus.RESERVED, RiskAuthorizationStatus.EXPIRED, RiskAuthorizationStatus.INVALIDATED,
            RiskAuthorizationStatus.REVOKED,
        ):
            assert is_legal_risk_authorization_status_transition(RiskAuthorizationStatus.ISSUED, target)

    def test_issued_cannot_transition_directly_to_consumed(self) -> None:
        # A reservation must be recorded before consumption -- see Phase 3's
        # "RESERVE AND CONSUME SEMANTICS".
        assert not is_legal_risk_authorization_status_transition(RiskAuthorizationStatus.ISSUED, RiskAuthorizationStatus.CONSUMED)

    def test_reserved_can_transition_to_consumed_or_a_terminal_status(self) -> None:
        for target in (
            RiskAuthorizationStatus.CONSUMED, RiskAuthorizationStatus.EXPIRED, RiskAuthorizationStatus.INVALIDATED,
            RiskAuthorizationStatus.REVOKED,
        ):
            assert is_legal_risk_authorization_status_transition(RiskAuthorizationStatus.RESERVED, target)

    def test_no_terminal_status_transitions_anywhere(self) -> None:
        for terminal in (
            RiskAuthorizationStatus.CONSUMED, RiskAuthorizationStatus.EXPIRED, RiskAuthorizationStatus.INVALIDATED,
            RiskAuthorizationStatus.REVOKED,
        ):
            for target in RiskAuthorizationStatus:
                assert not is_legal_risk_authorization_status_transition(terminal, target)

    def test_nothing_transitions_back_to_issued(self) -> None:
        for current in RiskAuthorizationStatus:
            assert not is_legal_risk_authorization_status_transition(current, RiskAuthorizationStatus.ISSUED)

    def test_nothing_transitions_back_to_reserved(self) -> None:
        for current in RiskAuthorizationStatus:
            if current is RiskAuthorizationStatus.ISSUED:
                continue
            assert not is_legal_risk_authorization_status_transition(current, RiskAuthorizationStatus.RESERVED)

    def test_terminal_classification(self) -> None:
        assert not is_terminal_risk_authorization_status(RiskAuthorizationStatus.ISSUED)
        assert not is_terminal_risk_authorization_status(RiskAuthorizationStatus.RESERVED)
        for terminal in (
            RiskAuthorizationStatus.CONSUMED, RiskAuthorizationStatus.EXPIRED, RiskAuthorizationStatus.INVALIDATED,
            RiskAuthorizationStatus.REVOKED,
        ):
            assert is_terminal_risk_authorization_status(terminal)
