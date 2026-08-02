"""Resolves a `bindings.MacroDatasetBinding` into the `value`/`release_time`
`pandas.DataFrame` shape `features.macro.macro_features.register_macro_features`
already expects as one entry of `FeatureContext.macro_data` (keyed by
`MacroSourceConfig.source_name`) -- reads Milestone 10 Phase 4B's curated
FRED layer (`collectors.curated.macro_observation.CuratedObservationStore`
+ `collectors.curated.datasets.ComponentDatasetManifestStore`).

REVISION POLICY IS A SELECTION, NOT A JOIN RULE. The actual point-in-time
join is entirely `features.alignment.as_of_join_external`'s job, keyed on
`release_time` (mapped 1:1 from `CuratedMacroObservation.availability_time`
-- Phase 4B's own release/availability proof, never `observation_date`).
This module's only job is choosing WHICH observations enter that join:

  * `RevisionPolicyKind.VINTAGE_SERIES` -- pass every distinct vintage
    through unchanged. Because each vintage carries its own accurate
    `availability_time`, `as_of_join_external`'s backward-looking as-of
    join over the FULL multi-vintage stream already reproduces true
    point-in-time-correct "most recently released value as of T" -- this
    is the general-purpose, always-safe default.
  * `RevisionPolicyKind.FIRST_RELEASE_ONLY` -- keep only the
    earliest-released vintage per `observation_date`, discarding every
    later revision even if the store holds one -- an explicit, narrower
    selection a researcher opts into (e.g. to study a strategy using only
    ever-first-published values), never the default.
  * `RevisionPolicyKind.LATEST_AVAILABLE` / `AS_OF_REALTIME_DATE` --
    REFUSED outright (`AlignmentPolicyError`) for research/training
    dataset construction. Both are explicitly non-point-in-time-safe by
    their own `market_data.collectors.curated.revision_policy` docstrings
    (a single global "whatever is true right now" or "whatever was true
    on one fixed historical date" snapshot, not a per-row as-of view) --
    this bridge never silently builds a leaky dataset from either.

A genuinely missing observation (`CuratedMacroObservation.is_missing=True`)
is NOT dropped from the stream -- it is included with `value=NaN` at its
own real `availability_time`, so the as-of join correctly shows the
series as unavailable/blank starting exactly when the official gap was
published, rather than silently carrying the prior value forward across
a genuine data gap (spec's "missing stays missing per policy; no future
backfill" requirement, achieved entirely through the EXISTING as-of join
semantics -- no new machinery needed).
"""

from __future__ import annotations

import pandas as pd

from quant_platform.core.exceptions import AlignmentPolicyError, SourceVerificationError
from quant_platform.features.market_data_bridge.bindings import MacroDatasetBinding
from quant_platform.market_data.collectors.curated.datasets import ComponentDatasetManifestStore
from quant_platform.market_data.collectors.curated.macro_observation import (
    CuratedMacroObservation,
    CuratedObservationStore,
)
from quant_platform.market_data.collectors.curated.revision_policy import RevisionPolicyKind
from quant_platform.market_data.identity import compute_content_id

__all__ = [
    "PIT_SAFE_REVISION_POLICY_KINDS",
    "resolve_macro_dataframe",
    "select_observations_for_policy",
    "verify_macro_binding",
]

PIT_SAFE_REVISION_POLICY_KINDS = frozenset({RevisionPolicyKind.VINTAGE_SERIES, RevisionPolicyKind.FIRST_RELEASE_ONLY})


def verify_macro_binding(
    observation_store: CuratedObservationStore, manifest_store: ComponentDatasetManifestStore, binding: MacroDatasetBinding,
) -> list[CuratedMacroObservation]:
    """Independently re-verifies `binding` against the CURRENT durable
    curated-macro repository state: the pinned `component_manifest_id`
    must equal the store's current manifest for `(provider, series_id)`,
    AND a live `CuratedObservationStore` read must reproduce that
    manifest's own `semantic_digest` (via the exact same
    `compute_content_id("curated_component_semantic_digest", ...)`
    formula `collectors.curated.datasets.create_component_dataset_manifest`
    itself uses -- reused, not reimplemented, so this proves "the live
    read matches what the durable manifest claims" rather than
    re-deriving a materially different hash of the same intent). Fails
    closed (`SourceVerificationError`) on any mismatch, exactly like
    `base_asset_adapter.verify_base_asset_binding`."""
    current = manifest_store.read_current(binding.provider, binding.series_id)
    if current is None:
        raise SourceVerificationError(
            f"No curated component manifest exists for provider={binding.provider!r} series_id={binding.series_id!r}",
            context={"provider": binding.provider, "series_id": binding.series_id},
        )
    if current.component_manifest_id != binding.component_manifest_id:
        raise SourceVerificationError(
            f"MacroDatasetBinding.component_manifest_id={binding.component_manifest_id!r} does not match the "
            f"CURRENT component manifest id={current.component_manifest_id!r} for "
            f"{binding.provider}/{binding.series_id} -- re-pin this binding to the current id as a deliberate, "
            "explicit action; this bridge never silently substitutes newer data for a stale pin.",
            context={"pinned": binding.component_manifest_id, "current": current.component_manifest_id},
        )
    observations = observation_store.read_observations(binding.provider, binding.series_id)
    recomputed_digest = compute_content_id(
        "curated_component_semantic_digest", {"observation_ids": sorted(o.observation_id for o in observations)}
    )
    if recomputed_digest != current.semantic_digest:
        raise SourceVerificationError(
            "Recomputed semantic digest of a live observation-store read does not match the current manifest's "
            "own recorded semantic_digest -- the durable observation store and its manifest have diverged; "
            "refusing to build a research dataset from unverifiable macro data.",
            context={
                "provider": binding.provider, "series_id": binding.series_id,
                "manifest_semantic_digest": current.semantic_digest, "recomputed_semantic_digest": recomputed_digest,
            },
        )
    if not observations:
        raise SourceVerificationError(
            f"Binding {binding.binding_id} resolved zero observations for {binding.provider}/{binding.series_id}",
            context={"provider": binding.provider, "series_id": binding.series_id},
        )
    return observations


def select_observations_for_policy(
    observations: list[CuratedMacroObservation], *, kind: RevisionPolicyKind,
) -> list[CuratedMacroObservation]:
    """Pure selection over an already-verified observation set -- see
    module docstring for exactly what each `RevisionPolicyKind` selects.
    Deterministic regardless of the input list's own order (sorts before
    grouping/selecting, never relies on dict/set iteration order)."""
    if kind not in PIT_SAFE_REVISION_POLICY_KINDS:
        raise AlignmentPolicyError(
            f"RevisionPolicyKind.{kind.name} is not point-in-time-safe for research/training dataset "
            "construction -- only VINTAGE_SERIES and FIRST_RELEASE_ONLY are permitted here. LATEST_AVAILABLE "
            "reflects whatever is true right now (not what was known historically at each row); "
            "AS_OF_REALTIME_DATE reflects one fixed global snapshot date, not a per-row as-of view. Refusing "
            "to silently build a leaky dataset.",
            context={"revision_policy_kind": kind.value},
        )
    ordered = sorted(
        observations, key=lambda o: (o.availability_time, o.observation_date, o.realtime_start or "", o.observation_id)
    )
    if kind is RevisionPolicyKind.VINTAGE_SERIES:
        return ordered
    # FIRST_RELEASE_ONLY: earliest-released vintage per observation_date.
    first_by_date: dict[str, CuratedMacroObservation] = {}
    for observation in ordered:
        existing = first_by_date.get(observation.observation_date)
        if existing is None or observation.availability_time < existing.availability_time:
            first_by_date[observation.observation_date] = observation
    return sorted(first_by_date.values(), key=lambda o: (o.availability_time, o.observation_date, o.observation_id))


def resolve_macro_dataframe(
    observation_store: CuratedObservationStore, manifest_store: ComponentDatasetManifestStore, binding: MacroDatasetBinding,
) -> pd.DataFrame:
    """Verified, deterministically ordered `value`/`release_time` frame
    for `binding` -- the single, documented Decimal -> float64 boundary
    crossing for macro data. `value` is `NaN` for a genuinely missing
    observation (see module docstring)."""
    observations = verify_macro_binding(observation_store, manifest_store, binding)
    selected = select_observations_for_policy(observations, kind=binding.revision_policy_kind)
    rows = [
        {
            "value": (float("nan") if o.is_missing else float(o.value)),  # type: ignore[arg-type]
            "release_time": pd.Timestamp(o.availability_time),
        }
        for o in selected
    ]
    df = pd.DataFrame(rows, columns=["value", "release_time"])
    df["release_time"] = pd.to_datetime(df["release_time"], utc=True)
    df = df.sort_values("release_time", kind="mergesort").reset_index(drop=True)
    return df
