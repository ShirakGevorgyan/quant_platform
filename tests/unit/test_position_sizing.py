"""Tests for position sizing / risk management."""

from __future__ import annotations

import pytest

from quant_platform.core.exceptions import RiskLimitExceededError
from quant_platform.risk.position_sizing import (
    FixedFractionalSizer,
    KellyCriterionSizer,
    VolatilityTargetSizer,
)


class TestFixedFractionalSizer:
    def test_computes_expected_quantity(self) -> None:
        sizer = FixedFractionalSizer(risk_percent=1.0)
        # Stop distance chosen wide enough that the result stays within the
        # default 1x max_position_fraction cap (tested separately below).
        quantity = sizer.size(
            account_equity=10_000.0, entry_price=2000.0, point_value=1.0, stop_loss_price=1900.0
        )
        # risk_amount = 100; stop_distance = 100; quantity = 100/100 = 1
        assert quantity == pytest.approx(1.0)

    def test_requires_stop_loss_price(self) -> None:
        sizer = FixedFractionalSizer(risk_percent=1.0)
        with pytest.raises(ValueError, match="requires stop_loss_price"):
            sizer.size(account_equity=10_000.0, entry_price=2000.0)

    def test_requires_stop_loss_different_from_entry(self) -> None:
        sizer = FixedFractionalSizer(risk_percent=1.0)
        with pytest.raises(ValueError, match="must differ"):
            sizer.size(account_equity=10_000.0, entry_price=2000.0, stop_loss_price=2000.0)

    def test_rejects_invalid_risk_percent(self) -> None:
        with pytest.raises(ValueError):
            FixedFractionalSizer(risk_percent=0.0)
        with pytest.raises(ValueError):
            FixedFractionalSizer(risk_percent=101.0)

    def test_clamps_to_max_position_fraction_by_default(self) -> None:
        # An extremely tight stop would otherwise produce huge leverage.
        sizer = FixedFractionalSizer(risk_percent=50.0, max_position_fraction=1.0)
        quantity = sizer.size(
            account_equity=10_000.0, entry_price=2000.0, point_value=1.0, stop_loss_price=1999.99
        )
        max_quantity = 10_000.0 / 2000.0
        assert quantity == pytest.approx(max_quantity)

    def test_raises_on_limit_exceeded_when_configured(self) -> None:
        sizer = FixedFractionalSizer(risk_percent=50.0, max_position_fraction=1.0, on_limit_exceeded="raise")
        with pytest.raises(RiskLimitExceededError):
            sizer.size(account_equity=10_000.0, entry_price=2000.0, point_value=1.0, stop_loss_price=1999.99)


class TestVolatilityTargetSizer:
    def test_computes_expected_quantity(self) -> None:
        sizer = VolatilityTargetSizer(target_volatility_percent=2.0)
        # Volatility chosen wide enough that the result stays within the
        # default 1x max_position_fraction cap (tested separately elsewhere).
        quantity = sizer.size(
            account_equity=10_000.0, entry_price=2000.0, point_value=1.0, current_volatility=200.0
        )
        # target amount = 200; quantity = 200/200 = 1
        assert quantity == pytest.approx(1.0)

    def test_requires_current_volatility(self) -> None:
        sizer = VolatilityTargetSizer(target_volatility_percent=2.0)
        with pytest.raises(ValueError, match="current_volatility"):
            sizer.size(account_equity=10_000.0, entry_price=2000.0)

    def test_rejects_non_positive_volatility(self) -> None:
        sizer = VolatilityTargetSizer(target_volatility_percent=2.0)
        with pytest.raises(ValueError):
            sizer.size(account_equity=10_000.0, entry_price=2000.0, current_volatility=0.0)


class TestKellyCriterionSizer:
    def test_full_kelly_formula(self) -> None:
        # win_rate=0.6, win_loss_ratio=2 -> full_kelly = 0.6 - 0.4/2 = 0.4
        sizer = KellyCriterionSizer(win_rate=0.6, win_loss_ratio=2.0, kelly_fraction=1.0)
        assert sizer.full_kelly == pytest.approx(0.4)

    def test_half_kelly_halves_the_allocation(self) -> None:
        full = KellyCriterionSizer(win_rate=0.6, win_loss_ratio=2.0, kelly_fraction=1.0)
        half = KellyCriterionSizer(win_rate=0.6, win_loss_ratio=2.0, kelly_fraction=0.5)
        q_full = full.size(account_equity=10_000.0, entry_price=100.0, point_value=1.0)
        q_half = half.size(account_equity=10_000.0, entry_price=100.0, point_value=1.0)
        assert q_half == pytest.approx(q_full / 2.0)

    def test_negative_edge_sizes_to_zero(self) -> None:
        # win_rate=0.3, win_loss_ratio=1 -> full_kelly = 0.3 - 0.7 = -0.4 (losing edge)
        sizer = KellyCriterionSizer(win_rate=0.3, win_loss_ratio=1.0)
        quantity = sizer.size(account_equity=10_000.0, entry_price=100.0, point_value=1.0)
        assert quantity == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("win_rate", "win_loss_ratio", "kelly_fraction"),
        [(0.0, 2.0, 0.5), (1.0, 2.0, 0.5), (0.6, 0.0, 0.5), (0.6, 2.0, 0.0), (0.6, 2.0, 1.5)],
    )
    def test_rejects_invalid_construction(
        self, win_rate: float, win_loss_ratio: float, kelly_fraction: float
    ) -> None:
        with pytest.raises(ValueError):
            KellyCriterionSizer(win_rate=win_rate, win_loss_ratio=win_loss_ratio, kelly_fraction=kelly_fraction)


class TestSharedValidation:
    def test_rejects_non_positive_equity(self) -> None:
        sizer = FixedFractionalSizer(risk_percent=1.0)
        with pytest.raises(ValueError, match="account_equity"):
            sizer.size(account_equity=0.0, entry_price=100.0, stop_loss_price=90.0)

    def test_rejects_non_positive_entry_price(self) -> None:
        sizer = FixedFractionalSizer(risk_percent=1.0)
        with pytest.raises(ValueError, match="entry_price"):
            sizer.size(account_equity=10_000.0, entry_price=0.0, stop_loss_price=-10.0)

    @pytest.mark.parametrize("max_fraction", [0.0, -0.5, 1.5])
    def test_rejects_invalid_max_position_fraction(self, max_fraction: float) -> None:
        with pytest.raises(ValueError):
            FixedFractionalSizer(risk_percent=1.0, max_position_fraction=max_fraction)

    def test_rejects_invalid_limit_policy(self) -> None:
        with pytest.raises(ValueError):
            FixedFractionalSizer(risk_percent=1.0, on_limit_exceeded="ignore")  # type: ignore[arg-type]
