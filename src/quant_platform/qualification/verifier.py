"""`QualificationVerifier` (Milestone 11, Phase 1): independent,
non-raising fact-finding over an already-built research dataset --
identity re-derivation, artifact readability/corruption, replay
cross-checks, lineage completeness, and required-feature presence. Every
`dimensions.py` evaluator CONSUMES these facts rather than re-deriving
them, so the same underlying check is never computed twice.

VERIFY, NEVER TRUST. Every function here re-reads or re-derives evidence
independently of whatever the manifest itself claims -- never a bare
`assert manifest.some_field`. Mirrors `features.market_data_bridge.
base_asset_adapter.verify_base_asset_binding`'s own "verify, then let the
caller decide what to do about it" shape, one layer up: this module
NEVER raises for an ordinary corruption/mismatch finding (that is always
reported as a `VerificationFacts` field for `dimensions.py`/`engine.py`
to turn into a `BlockingFailure`), and raises `QualificationVerificationError`
only when a check cannot even be ATTEMPTED (e.g. a filesystem permission
error unrelated to corruption)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    QualificationVerificationError,
    ResearchDatasetError,
)
from quant_platform.features.manifests import (
    ResearchDatasetManifest,
    ResearchDatasetStore,
    compute_dataset_id,
)
from quant_platform.features.validation import DatasetIssueType, validate_research_dataset
from quant_platform.ml.persistence import read_json_file

__all__ = [
    "ArtifactReadResult",
    "QualificationVerifier",
    "VerificationFacts",
    "verify_artifacts",
    "verify_identity",
    "verify_lineage",
    "verify_no_future_leakage",
    "verify_required_features",
]

_METADATA_FILE_NAME = "metadata.json"
"""Matches `features.manifests._METADATA_FILE` exactly (that constant is
module-private, not exported) -- `write_artifacts` writes this file into
every content-addressed directory alongside the split Parquet files."""

_LINEAGE_FIELDS: tuple[str, ...] = ("source_historical_dataset_id", "source_historical_manifest_version", "code_revision")
"""Required PROVENANCE fields on `ResearchDatasetManifest` itself --
`input_content_hashes` (a dict, checked for non-emptiness separately) is
the 4th required lineage signal."""


def verify_identity(manifest: ResearchDatasetManifest) -> tuple[bool, str | None]:
    """Recomputes `features.manifests.compute_dataset_id` from the
    manifest's own recipe fields (symbol/timeframe/feature registry
    fingerprint/label/split/preprocessing definitions) and requires an
    exact match against `manifest.dataset_id` -- the SAME function
    `dataset_builder.ResearchDatasetBuilder.build` itself uses to mint
    it, reused here rather than reimplemented."""
    recomputed = compute_dataset_id(
        symbol=manifest.symbol, base_timeframe=manifest.base_timeframe, feature_registry_fingerprint=manifest.feature_registry_fingerprint,
        label_definition=manifest.label_definition, split_definition=manifest.split_definition,
        preprocessing_definition=manifest.preprocessing_definition,
    )
    if recomputed == manifest.dataset_id:
        return True, None
    return False, f"recomputed dataset_id {recomputed!r} does not match manifest.dataset_id {manifest.dataset_id!r}"


@dataclass(frozen=True, slots=True)
class ArtifactReadResult:
    readable: bool
    error_message: str | None
    splits: dict[str, pd.DataFrame] | None
    metadata_checksums_match_manifest: bool
    row_counts_match_manifest: bool


def _read_metadata_json(research_store: ResearchDatasetStore, dataset_id: str, content_id: str) -> dict[str, object] | None:
    path = research_store.content_dir(dataset_id, content_id) / _METADATA_FILE_NAME
    if not path.is_file():
        return None
    try:
        decoded = read_json_file(path)
    except (ArtifactCorruptionError, OSError):
        return None  # a read/parse failure here means "cannot cross-check" -- reported by the caller via metadata_checksums_match_manifest=False, never raised
    return decoded if isinstance(decoded, dict) else None


def verify_artifacts(research_store: ResearchDatasetStore, manifest: ResearchDatasetManifest) -> ArtifactReadResult:
    """Reads the manifest's own pinned `content_id` via `ResearchDatasetStore.
    read_artifacts` (which ALREADY re-verifies every split's own Parquet
    file against `metadata.json`'s recorded checksum internally, raising
    `ResearchDatasetError` on any mismatch -- caught here and reported as
    `readable=False`, never propagated) -- then performs one ADDITIONAL,
    independent cross-check `read_artifacts` does not itself make:
    whether the manifest's OWN separately-stored `output_content_hashes`/
    `row_counts` fields agree with `metadata.json`'s own recorded values.
    Two independently-persisted records disagreeing (the manifest JSON
    file vs. the content directory's own metadata.json) is exactly a
    REPLAY_MISMATCH signal -- a tamper/divergence a single file's own
    internal checksum check cannot catch."""
    try:
        splits = research_store.read_artifacts(manifest.dataset_id, manifest.content_id)
    except ResearchDatasetError as exc:
        return ArtifactReadResult(readable=False, error_message=str(exc), splits=None, metadata_checksums_match_manifest=False, row_counts_match_manifest=False)
    if splits is None:
        return ArtifactReadResult(
            readable=False, error_message=f"no content directory exists for dataset_id={manifest.dataset_id!r} content_id={manifest.content_id!r}",
            splits=None, metadata_checksums_match_manifest=False, row_counts_match_manifest=False,
        )

    metadata = _read_metadata_json(research_store, manifest.dataset_id, manifest.content_id)
    if metadata is None:
        return ArtifactReadResult(readable=True, error_message=None, splits=splits, metadata_checksums_match_manifest=False, row_counts_match_manifest=False)

    recorded_checksums = metadata.get("per_split_checksums")
    recorded_row_counts = metadata.get("row_counts")
    checksums_match = recorded_checksums == manifest.output_content_hashes
    row_counts_match = recorded_row_counts == manifest.row_counts
    return ArtifactReadResult(
        readable=True, error_message=None, splits=splits, metadata_checksums_match_manifest=checksums_match, row_counts_match_manifest=row_counts_match,
    )


def verify_lineage(manifest: ResearchDatasetManifest) -> tuple[bool, tuple[str, ...]]:
    missing = [name for name in _LINEAGE_FIELDS if not getattr(manifest, name)]
    if not manifest.input_content_hashes:
        missing.append("input_content_hashes")
    return (not missing, tuple(missing))


def verify_required_features(manifest: ResearchDatasetManifest, required_feature_names: frozenset[str]) -> tuple[bool, tuple[str, ...]]:
    present = set(manifest.feature_names)
    missing = tuple(sorted(name for name in required_feature_names if name not in present))
    return (not missing, missing)


def verify_no_future_leakage(split_df: pd.DataFrame, *, split_name: str) -> tuple[bool, tuple[str, ...]]:
    """Runs the REAL, unmodified `features.validation.validate_research_dataset`
    over one split's own stored artifact (never a second leakage-detection
    implementation) and extracts only its `TARGET_LEAKAGE_SUSPECTED`
    finding(s) -- statistical/missingness findings from the SAME call are
    consumed separately by the Statistical Integrity dimension, not here."""
    if "open_time" not in split_df.columns:
        return True, ()  # a split artifact missing open_time entirely is a STRUCTURAL finding, not this function's concern
    feature_columns = [c for c in split_df.columns if c not in ("open_time", "label", "label_valid")]
    timestamps = split_df["open_time"]
    labels = split_df["label"] if "label" in split_df.columns else None
    report = validate_research_dataset(split_df[feature_columns], timestamps=timestamps, labels=labels)
    leakage_issues = [i for i in report.issues if i.issue_type is DatasetIssueType.TARGET_LEAKAGE_SUSPECTED]
    if not leakage_issues:
        return True, ()
    return False, tuple(f"{split_name}: {issue.message}" for issue in leakage_issues)


@dataclass(frozen=True, slots=True)
class VerificationFacts:
    """The complete bundle `dimensions.py`'s evaluators and `engine.py`
    consume -- computed exactly ONCE per qualification run."""

    identity_matches: bool
    identity_message: str | None
    artifacts: ArtifactReadResult
    lineage_present: bool
    missing_lineage_fields: tuple[str, ...]
    required_features_present: bool
    missing_required_features: tuple[str, ...]
    leakage_free: bool
    leakage_messages: tuple[str, ...] = field(default_factory=tuple)


class QualificationVerifier:
    """Bundles every free function above into one call,
    `verify(manifest, research_store, required_feature_names=...)`, so
    `DatasetQualificationEngine` (and any other caller) invokes the whole
    independent-verification pass exactly once."""

    def verify(
        self, manifest: ResearchDatasetManifest, research_store: ResearchDatasetStore, *, required_feature_names: frozenset[str] = frozenset(),
    ) -> VerificationFacts:
        try:
            identity_matches, identity_message = verify_identity(manifest)
            artifacts = verify_artifacts(research_store, manifest)
            lineage_present, missing_lineage_fields = verify_lineage(manifest)
            required_features_present, missing_required_features = verify_required_features(manifest, required_feature_names)

            leakage_free = True
            leakage_messages: tuple[str, ...] = ()
            if artifacts.readable and artifacts.splits:
                per_split_results = [verify_no_future_leakage(df, split_name=name) for name, df in sorted(artifacts.splits.items())]
                leakage_free = all(ok for ok, _ in per_split_results)
                leakage_messages = tuple(msg for _ok, msgs in per_split_results for msg in msgs)
        except QualificationVerificationError:
            raise
        except Exception as exc:
            raise QualificationVerificationError(
                f"QualificationVerifier.verify could not complete for dataset_id={manifest.dataset_id!r}: {exc}",
                context={"dataset_id": manifest.dataset_id, "version": manifest.version},
            ) from exc

        return VerificationFacts(
            identity_matches=identity_matches, identity_message=identity_message, artifacts=artifacts, lineage_present=lineage_present,
            missing_lineage_fields=missing_lineage_fields, required_features_present=required_features_present,
            missing_required_features=missing_required_features, leakage_free=leakage_free, leakage_messages=leakage_messages,
        )
