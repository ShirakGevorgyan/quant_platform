"""Unit tests for `portfolio_risk.identity`: the shared Decimal<->JSON
and content-id primitives every other model in this package builds on."""

from __future__ import annotations

from decimal import Decimal

import pytest

from quant_platform.core.exceptions import PortfolioRiskPolicyError
from quant_platform.portfolio_risk.identity import (
    compute_content_id,
    decimal_from_float,
    decimal_to_json,
    is_valid_sha256_hex,
    parse_decimal,
)


class TestDecimalToJson:
    def test_round_trips_exactly(self) -> None:
        assert decimal_to_json(Decimal("1.10")) == "1.10"

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(PortfolioRiskPolicyError):
            decimal_to_json(Decimal("NaN"))
        with pytest.raises(PortfolioRiskPolicyError):
            decimal_to_json(Decimal("Infinity"))


class TestDecimalFromFloat:
    def test_never_reproduces_binary_float_imprecision(self) -> None:
        # Decimal(0.1) != Decimal("0.1") -- decimal_from_float must go
        # through str() to avoid this.
        assert decimal_from_float(0.1, field_name="x") == Decimal("0.1")
        assert decimal_from_float(0.1, field_name="x") != Decimal(0.1)  # noqa: RUF032 -- deliberately demonstrating the imprecision this function avoids

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(PortfolioRiskPolicyError):
            decimal_from_float(float("nan"), field_name="x")
        with pytest.raises(PortfolioRiskPolicyError):
            decimal_from_float(float("inf"), field_name="x")


class TestParseDecimal:
    def test_accepts_str_int_decimal(self) -> None:
        assert parse_decimal("1.5", field_name="x") == Decimal("1.5")
        assert parse_decimal(5, field_name="x") == Decimal(5)
        assert parse_decimal(Decimal("2.5"), field_name="x") == Decimal("2.5")

    def test_rejects_bool(self) -> None:
        with pytest.raises(PortfolioRiskPolicyError):
            parse_decimal(True, field_name="x")

    def test_rejects_float(self) -> None:
        # float is deliberately NOT accepted -- callers must go through
        # decimal_from_float explicitly so binary-float imprecision is
        # never silently laundered through parse_decimal.
        with pytest.raises(PortfolioRiskPolicyError):
            parse_decimal(1.5, field_name="x")

    def test_rejects_malformed_string(self) -> None:
        with pytest.raises(PortfolioRiskPolicyError):
            parse_decimal("not-a-number", field_name="x")

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(PortfolioRiskPolicyError):
            parse_decimal("NaN", field_name="x")


class TestComputeContentId:
    def test_deterministic(self) -> None:
        a = compute_content_id("kind", {"a": 1})
        b = compute_content_id("kind", {"a": 1})
        assert a == b
        assert is_valid_sha256_hex(a)

    def test_kind_participates_in_identity(self) -> None:
        a = compute_content_id("kind_a", {"a": 1})
        b = compute_content_id("kind_b", {"a": 1})
        assert a != b

    def test_payload_participates_in_identity(self) -> None:
        a = compute_content_id("kind", {"a": 1})
        b = compute_content_id("kind", {"a": 2})
        assert a != b
