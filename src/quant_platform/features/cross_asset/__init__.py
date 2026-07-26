"""Cross-asset feature group -- another instrument's returns/momentum/
volatility/correlation, backward-aligned into the base symbol's timeframe."""

from __future__ import annotations

from quant_platform.features.cross_asset.cross_asset import CrossAssetWindows, register_cross_asset_features

__all__ = ["CrossAssetWindows", "register_cross_asset_features"]
