"""Milestone 3 -- leak-free feature engineering and research dataset
platform. See `docs/feature_engineering.md` for the full architecture and
`examples/xauusd_research_dataset.example.json` for a worked configuration.
"""

from __future__ import annotations

FEATURE_ENGINE_VERSION = "1.0.0"
"""Bumped whenever this package's own computation semantics change in a way
that could alter feature values for an unchanged `FeatureSpec` (e.g. a bug
fix in a shared rolling-window helper). Recorded in every research dataset
manifest (`manifests.ResearchDatasetManifest.feature_engine_version`) so a
manifest is traceable to the exact engine behavior that produced it, on top
of the per-feature version already carried by `FeatureSpec`."""

__all__ = ["FEATURE_ENGINE_VERSION"]
