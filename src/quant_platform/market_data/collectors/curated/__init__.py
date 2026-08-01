"""Curated FRED macro universe (Milestone 10, Phase 4B) -- built ON TOP
OF, never bypassing, Phase 4A's secure FRED collector infrastructure.
Adds: a versioned curated series registry (`registry.py`), strict
release/vintage point-in-time policy (`revision_policy.py`/
`availability.py`), official-metadata drift verification (`metadata.py`),
a richer canonical observation model (`macro_observation.py`), a
multi-series backfill spec (`backfill.py`), per-series/combined dataset
manifests (`datasets.py`), 12-stage multi-series orchestration
(`orchestration.py`), pure incremental update planning (`update_plan.py`),
reconciliation/verification/reports, and an opt-in real-FRED acceptance
workflow (`acceptance.py`, the ONLY place here that reads an
environment variable).

Still HTTPS-only, still `api.stlouisfed.org`-only, still never persists
a credential -- every security/secret-handling guarantee Phase 4A
established applies unchanged to everything in this subpackage, since
it reuses Phase 4A's transport/retry/rate-limit/cache/request-manifest/
response-manifest machinery directly rather than reimplementing any of
it."""

from __future__ import annotations
