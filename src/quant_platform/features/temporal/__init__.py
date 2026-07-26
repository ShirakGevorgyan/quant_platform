"""Temporal feature group -- hour/day/month, session flags, and cyclical
encodings, computed purely from a bar's own `open_time`."""

from __future__ import annotations

from quant_platform.features.temporal.calendar_features import register_core_temporal_features

__all__ = ["register_core_temporal_features"]
