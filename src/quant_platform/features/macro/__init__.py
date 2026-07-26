"""Macro/external-data feature group -- rate/yield levels, changes,
release-age, and staleness flags, joined strictly by release timestamp."""

from __future__ import annotations

from quant_platform.features.macro.macro_features import MacroSourceConfig, register_macro_features

__all__ = ["MacroSourceConfig", "register_macro_features"]
