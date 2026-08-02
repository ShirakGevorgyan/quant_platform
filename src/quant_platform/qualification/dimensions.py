"""The 8 qualification dimension evaluators (Milestone 11, Phase 1,
spec's own named list). Each is a pure function consuming the already-
built `ResearchDatasetManifest`, its read split `DataFrame`s, and the
`verifier.VerificationFacts` computed ONCE by `QualificationVerifier` --
never re-deriving a fact a dimension shares with another (e.g. both
Structural Integrity and Temporal Integrity report on `facts`, neither
re-reads the artifact store itself).

BLOCKING-FAILURE-TO-DIMENSION MAPPING (the 6 named codes, each owned by
exactly one dimension -- documented once, here, rather than scattered):

    FUTURE_LEAKAGE            -> Temporal Integrity
    MANIFEST_CORRUPTION       -> Structural Integrity
    REPLAY_MISMATCH           -> Determinism
    IDENTITY_MISMATCH         -> Reproducibility
    MISSING_LINEAGE           -> Reproducibility
    REQUIRED_FEATURE_MISSING  -> Structural Integrity

Every OTHER finding this module makes (statistical/coverage/stability/
safety concerns) -- however severe -- becomes a `finding`/`warning`
only, never a `BlockingFailure`: the spec names exactly 6 blocking
failure codes, and this module never invents a 7th.

Reuses (never reimplements): `features.validation.validate_research_dataset`
(statistical/missingness/leakage findings), `features.drift.compare_splits`
(Stability's own PSI/shift/correlation analysis)."""

from __future__ import annotations

import pandas as pd

from quant_platform.features.drift import compare_splits
from quant_platform.features.manifests import ResearchDatasetManifest
from quant_platform.features.validation import DatasetIssueType, validate_research_dataset
from quant_platform.historical.quality import Severity
from quant_platform.qualification.models import (
    BlockingFailure,
    BlockingFailureCode,
    DimensionResult,
    QualificationDimensionKind,
)
from quant_platform.qualification.verifier import VerificationFacts

__all__ = [
    "evaluate_coverage",
    "evaluate_determinism",
    "evaluate_reproducibility",
    "evaluate_safety",
    "evaluate_stability",
    "evaluate_statistical_integrity",
    "evaluate_structural_integrity",
    "evaluate_temporal_integrity",
]

_NON_FEATURE_COLUMNS = frozenset({"open_time", "label", "label_valid"})
_RESERVED_LABEL_PREFIXES = ("label_", "target_")
"""Matches `features.engine._RESERVED_PREFIXES` exactly -- a feature
column with this prefix would mean label information leaked into the
feature space at build time; re-checked independently here on the
STORED artifact, not merely trusted from the build-time guard."""


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_FEATURE_COLUMNS]


# --------------------------------------------------------------------------
# 1. Structural Integrity -- MANIFEST_CORRUPTION, REQUIRED_FEATURE_MISSING
# --------------------------------------------------------------------------
def evaluate_structural_integrity(
    manifest: ResearchDatasetManifest, splits: dict[str, pd.DataFrame] | None, facts: VerificationFacts,
) -> DimensionResult:
    findings: list[str] = []
    warnings: list[str] = []
    blocking: list[BlockingFailure] = []
    recommendations: list[str] = []

    if not facts.artifacts.readable:
        blocking.append(
            BlockingFailure(
                code=BlockingFailureCode.MANIFEST_CORRUPTION, dimension=QualificationDimensionKind.STRUCTURAL_INTEGRITY,
                message=f"research dataset artifacts are unreadable or corrupted: {facts.artifacts.error_message}",
                context={"dataset_id": manifest.dataset_id, "content_id": manifest.content_id},
            )
        )
        recommendations.append("Rebuild this research dataset from source -- the stored content directory is missing or fails its own checksum verification.")
        return DimensionResult(
            dimension=QualificationDimensionKind.STRUCTURAL_INTEGRITY, score=0.0, findings=tuple(findings), warnings=tuple(warnings),
            blocking_failures=tuple(blocking), recommendations=tuple(recommendations),
        )

    assert splits is not None
    findings.append(f"{len(splits)} split(s) present: {sorted(splits)}")

    expected_feature_names = set(manifest.feature_names)
    for split_name, df in sorted(splits.items()):
        if len(df) == 0:
            warnings.append(f"split {split_name!r} is empty (0 rows)")
        present = set(_feature_columns(df))
        missing_from_split = sorted(expected_feature_names - present)
        if missing_from_split:
            warnings.append(f"split {split_name!r} is missing declared feature column(s): {missing_from_split}")
        if "label" not in df.columns:
            warnings.append(f"split {split_name!r} has no 'label' column")
        non_numeric = [c for c in _feature_columns(df) if c in expected_feature_names and not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            warnings.append(f"split {split_name!r} has non-numeric declared feature column(s): {non_numeric}")

    if not facts.artifacts.row_counts_match_manifest:
        warnings.append("manifest.row_counts does not match the content store's own recorded metadata.json row_counts")

    if not facts.required_features_present:
        blocking.append(
            BlockingFailure(
                code=BlockingFailureCode.REQUIRED_FEATURE_MISSING, dimension=QualificationDimensionKind.STRUCTURAL_INTEGRITY,
                message=f"required feature(s) not present in this dataset: {list(facts.missing_required_features)}",
                context={"missing_required_features": list(facts.missing_required_features)},
            )
        )
        recommendations.append("Rebuild with a feature registry/request that includes every required feature name before using this dataset.")

    score = 1.0 if not (warnings or blocking) else _clamp_score(1.0 - 0.15 * len(warnings))
    if blocking:
        score = 0.0
    return DimensionResult(
        dimension=QualificationDimensionKind.STRUCTURAL_INTEGRITY, score=score, findings=tuple(findings), warnings=tuple(warnings),
        blocking_failures=tuple(blocking), recommendations=tuple(recommendations),
    )


# --------------------------------------------------------------------------
# 2. Temporal Integrity -- FUTURE_LEAKAGE
# --------------------------------------------------------------------------
_RECOGNIZED_TRAIN_EVAL_PAIRS: tuple[tuple[str, str], ...] = (("train", "validation"), ("train", "test"))


def _fold_group(split_name: str) -> str | None:
    if split_name in ("train", "validation", "test"):
        return None  # the chronological, non-fold-grouped case
    parts = split_name.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else split_name


def evaluate_temporal_integrity(
    manifest: ResearchDatasetManifest, splits: dict[str, pd.DataFrame] | None, facts: VerificationFacts,  # noqa: ARG001
) -> DimensionResult:
    findings: list[str] = []
    warnings: list[str] = []
    blocking: list[BlockingFailure] = []
    recommendations: list[str] = []

    if splits is None:
        return DimensionResult(dimension=QualificationDimensionKind.TEMPORAL_INTEGRITY, score=0.0, warnings=("artifacts unreadable -- temporal checks skipped",))

    for split_name, df in sorted(splits.items()):
        if "open_time" not in df.columns or len(df) == 0:
            continue
        ts = df["open_time"]
        if not ts.is_monotonic_increasing:
            warnings.append(f"split {split_name!r}: open_time is not monotonically increasing")
        if ts.duplicated().any():
            warnings.append(f"split {split_name!r}: open_time contains {int(ts.duplicated().sum())} duplicate value(s)")
    findings.append(f"checked open_time ordering across {len(splits or {})} split(s)")

    # Chronological train-before-eval ordering, scoped to recognized pairs
    # only (never comparing DIFFERENT walk-forward folds' train/test against
    # each other -- an earlier fold's test rows legitimately becoming a
    # later fold's train rows is expected, correct behavior, not leakage).
    for train_name, eval_name in _RECOGNIZED_TRAIN_EVAL_PAIRS:
        if train_name in splits and eval_name in splits and len(splits[train_name]) and len(splits[eval_name]):
            train_max = splits[train_name]["open_time"].max()
            eval_min = splits[eval_name]["open_time"].min()
            if train_max > eval_min:
                warnings.append(f"{train_name!r} open_time extends past the start of {eval_name!r} (train_max={train_max} > {eval_name}_min={eval_min})")

    fold_groups: dict[str, dict[str, pd.DataFrame]] = {}
    for split_name, df in splits.items():
        group = _fold_group(split_name)
        if group is None:
            continue
        fold_groups.setdefault(group, {})[split_name.rsplit("_", 1)[-1]] = df
    for group, members in sorted(fold_groups.items()):
        if "train" in members and "test" in members and len(members["train"]) and len(members["test"]):
            train_max = members["train"]["open_time"].max()
            test_min = members["test"]["open_time"].min()
            if train_max > test_min:
                warnings.append(f"fold group {group!r}: train open_time extends past the start of its own test split")

    if not facts.leakage_free:
        blocking.append(
            BlockingFailure(
                code=BlockingFailureCode.FUTURE_LEAKAGE, dimension=QualificationDimensionKind.TEMPORAL_INTEGRITY,
                message=f"suspected target leakage detected: {list(facts.leakage_messages)}",
                context={"leakage_messages": list(facts.leakage_messages)},
            )
        )
        recommendations.append("Investigate the flagged feature(s) for direct or near-direct encoding of the label; rebuild the dataset once resolved.")

    score = 0.0 if blocking else _clamp_score(1.0 - 0.2 * len(warnings))
    return DimensionResult(
        dimension=QualificationDimensionKind.TEMPORAL_INTEGRITY, score=score, findings=tuple(findings), warnings=tuple(warnings),
        blocking_failures=tuple(blocking), recommendations=tuple(recommendations),
    )


# --------------------------------------------------------------------------
# 3. Statistical Integrity -- never blocking (see module docstring)
# --------------------------------------------------------------------------
def evaluate_statistical_integrity(
    manifest: ResearchDatasetManifest, splits: dict[str, pd.DataFrame] | None,  # noqa: ARG001
) -> DimensionResult:
    if not splits or "train" not in splits or len(splits["train"]) == 0:
        return DimensionResult(dimension=QualificationDimensionKind.STATISTICAL_INTEGRITY, score=0.0, warnings=("no non-empty 'train' split available -- statistical checks skipped",))

    train_df = splits["train"]
    feature_columns = _feature_columns(train_df)
    report = validate_research_dataset(train_df[feature_columns], timestamps=train_df["open_time"] if "open_time" in train_df.columns else pd.Series(range(len(train_df))))

    findings = [f"validated {len(feature_columns)} feature column(s) over {len(train_df)} train row(s)"]
    warnings: list[str] = []
    for issue in report.issues:
        if issue.issue_type is DatasetIssueType.TARGET_LEAKAGE_SUSPECTED:
            continue  # owned by Temporal Integrity, not repeated here
        text = f"[{issue.severity.value}] {issue.issue_type.value}: {issue.message}"
        warnings.append(text)

    critical_count = sum(1 for i in report.issues if i.severity is Severity.CRITICAL and i.issue_type is not DatasetIssueType.TARGET_LEAKAGE_SUSPECTED)
    warning_count = sum(1 for i in report.issues if i.severity is Severity.WARNING)
    score = _clamp_score(1.0 - 0.3 * critical_count - 0.1 * warning_count)
    recommendations = ["Investigate CRITICAL-severity statistical findings before training a model on this dataset."] if critical_count else []
    return DimensionResult(
        dimension=QualificationDimensionKind.STATISTICAL_INTEGRITY, score=score, findings=tuple(findings), warnings=tuple(warnings),
        recommendations=tuple(recommendations),
    )


# --------------------------------------------------------------------------
# 4. Coverage -- never blocking
# --------------------------------------------------------------------------
def evaluate_coverage(manifest: ResearchDatasetManifest, splits: dict[str, pd.DataFrame] | None) -> DimensionResult:
    if not splits:
        return DimensionResult(dimension=QualificationDimensionKind.COVERAGE, score=0.0, warnings=("artifacts unreadable -- coverage checks skipped",))

    all_open_times = pd.concat([df["open_time"] for df in splits.values() if "open_time" in df.columns and len(df)], ignore_index=True) if any(len(df) for df in splits.values()) else pd.Series([], dtype="datetime64[ns, UTC]")
    findings: list[str] = []
    warnings: list[str] = []
    if len(all_open_times) == 0:
        warnings.append("no rows with an open_time column across any split")
        score = 0.0
    else:
        observed_start, observed_end = pd.Timestamp(all_open_times.min()), pd.Timestamp(all_open_times.max())
        requested_span = (manifest.utc_end - manifest.utc_start).total_seconds()
        observed_span = (observed_end - observed_start).total_seconds()
        coverage_fraction = _clamp_score(observed_span / requested_span) if requested_span > 0 else 1.0
        findings.append(f"requested=[{manifest.utc_start}, {manifest.utc_end}) observed=[{observed_start}, {observed_end}] coverage_fraction={coverage_fraction:.3f}")
        if coverage_fraction < 0.5:
            warnings.append(f"observed date coverage ({coverage_fraction:.1%}) is less than half of the requested range -- likely heavy warm-up/label trimming")

        total_rows = sum(len(df) for df in splits.values())
        feature_columns = sorted({c for df in splits.values() for c in _feature_columns(df)})
        null_fractions = {}
        for col in feature_columns:
            present_count = sum(len(df) for df in splits.values() if col in df.columns)
            null_count = sum(int(df[col].isna().sum()) for df in splits.values() if col in df.columns)
            null_fractions[col] = (null_count / present_count) if present_count else 1.0
        worst_column = max(null_fractions, key=lambda c: null_fractions[c]) if null_fractions else None
        if worst_column is not None and null_fractions[worst_column] > 0.5:
            warnings.append(f"feature {worst_column!r} is null in {null_fractions[worst_column]:.1%} of rows across all splits")
        findings.append(f"{total_rows} total row(s) across {len(splits)} split(s), {len(feature_columns)} feature column(s)")
        score = _clamp_score(coverage_fraction - 0.1 * len(warnings))

    return DimensionResult(dimension=QualificationDimensionKind.COVERAGE, score=score, findings=tuple(findings), warnings=tuple(warnings))


# --------------------------------------------------------------------------
# 5. Stability -- never blocking
# --------------------------------------------------------------------------
def evaluate_stability(manifest: ResearchDatasetManifest, splits: dict[str, pd.DataFrame] | None) -> DimensionResult:  # noqa: ARG001
    if not splits or "train" not in splits or len(splits["train"]) == 0:
        return DimensionResult(dimension=QualificationDimensionKind.STABILITY, score=0.0, warnings=("no non-empty 'train' split available -- stability checks skipped",))

    train_df = splits["train"]
    findings: list[str] = []
    warnings: list[str] = []
    psi_values: list[float] = []
    for comparison_name in ("validation", "test"):
        if comparison_name not in splits or len(splits[comparison_name]) == 0:
            continue
        drift = compare_splits(train_df, splits[comparison_name], reference_name="train", comparison_name=comparison_name)
        for fr in drift.feature_reports:
            if fr.population_stability_index == fr.population_stability_index:  # not NaN
                psi_values.append(fr.population_stability_index)
                if fr.population_stability_index >= 0.25:
                    warnings.append(f"{comparison_name}: feature {fr.feature_name!r} PSI={fr.population_stability_index:.3f} (>= 0.25, substantial distribution shift)")
        if drift.constant_features:
            warnings.append(f"{comparison_name}: constant feature(s) in train: {list(drift.constant_features)}")
        findings.append(f"compared train vs {comparison_name}: {len(drift.feature_reports)} shared numeric feature(s)")

    if not psi_values:
        return DimensionResult(dimension=QualificationDimensionKind.STABILITY, score=1.0, findings=tuple(findings) or ("only one non-empty split available -- no cross-split comparison possible",))

    mean_abs_psi = sum(abs(v) for v in psi_values) / len(psi_values)
    score = _clamp_score(1.0 - mean_abs_psi)
    recommendations = ["Investigate features with PSI >= 0.25 for a genuine train/eval distribution shift before relying on this dataset for model selection."] if any(v >= 0.25 for v in psi_values) else []
    return DimensionResult(dimension=QualificationDimensionKind.STABILITY, score=score, findings=tuple(findings), warnings=tuple(warnings), recommendations=tuple(recommendations))


# --------------------------------------------------------------------------
# 6. Determinism -- REPLAY_MISMATCH
# --------------------------------------------------------------------------
def evaluate_determinism(manifest: ResearchDatasetManifest, facts: VerificationFacts) -> DimensionResult:
    findings: list[str] = []
    blocking: list[BlockingFailure] = []
    recommendations: list[str] = []

    if not facts.artifacts.readable:
        return DimensionResult(dimension=QualificationDimensionKind.DETERMINISM, score=0.0, warnings=("artifacts unreadable -- determinism checks skipped (see Structural Integrity)",))

    if not facts.artifacts.metadata_checksums_match_manifest or not facts.artifacts.row_counts_match_manifest:
        blocking.append(
            BlockingFailure(
                code=BlockingFailureCode.REPLAY_MISMATCH, dimension=QualificationDimensionKind.DETERMINISM,
                message="manifest.output_content_hashes/row_counts do not match the content store's own independently-recorded metadata.json -- the two persisted records have diverged.",
                context={"dataset_id": manifest.dataset_id, "content_id": manifest.content_id},
            )
        )
        recommendations.append("Treat this dataset version as untrustworthy -- re-derive it from the manifest's own recorded lineage and re-save both records together.")
    else:
        findings.append("manifest.output_content_hashes and row_counts agree with the content store's own independently-recorded metadata.json")

    score = 0.0 if blocking else 1.0
    return DimensionResult(dimension=QualificationDimensionKind.DETERMINISM, score=score, findings=tuple(findings), blocking_failures=tuple(blocking), recommendations=tuple(recommendations))


# --------------------------------------------------------------------------
# 7. Reproducibility -- IDENTITY_MISMATCH, MISSING_LINEAGE
# --------------------------------------------------------------------------
def evaluate_reproducibility(manifest: ResearchDatasetManifest, facts: VerificationFacts) -> DimensionResult:
    findings: list[str] = []
    warnings: list[str] = []
    blocking: list[BlockingFailure] = []
    recommendations: list[str] = []

    if facts.identity_matches:
        findings.append("dataset_id is exactly reproducible from the manifest's own recipe fields (compute_dataset_id)")
    else:
        blocking.append(
            BlockingFailure(
                code=BlockingFailureCode.IDENTITY_MISMATCH, dimension=QualificationDimensionKind.REPRODUCIBILITY,
                message=facts.identity_message or "dataset_id does not match its own recomputed identity", context={"dataset_id": manifest.dataset_id},
            )
        )
        recommendations.append("Treat dataset_id as untrustworthy -- the manifest's own recipe fields no longer reproduce it; the manifest may have been tampered with or corrupted.")

    if facts.lineage_present:
        findings.append("all required provenance fields (source dataset id/version, code_revision, input_content_hashes) are present")
    else:
        blocking.append(
            BlockingFailure(
                code=BlockingFailureCode.MISSING_LINEAGE, dimension=QualificationDimensionKind.REPRODUCIBILITY,
                message=f"required provenance field(s) missing: {list(facts.missing_lineage_fields)}", context={"missing_lineage_fields": list(facts.missing_lineage_fields)},
            )
        )
        recommendations.append("Rebuild through the standard ResearchDatasetBuilder path, which always populates full provenance -- never hand-construct a manifest.")

    if not manifest.environment:
        warnings.append("manifest.environment metadata is empty -- reproduction under a different pandas/numpy/pyarrow version cannot be cross-checked")

    score = 0.0 if blocking else _clamp_score(1.0 - 0.1 * len(warnings))
    return DimensionResult(
        dimension=QualificationDimensionKind.REPRODUCIBILITY, score=score, findings=tuple(findings), warnings=tuple(warnings),
        blocking_failures=tuple(blocking), recommendations=tuple(recommendations),
    )


# --------------------------------------------------------------------------
# 8. Safety -- never blocking
# --------------------------------------------------------------------------
def evaluate_safety(manifest: ResearchDatasetManifest, splits: dict[str, pd.DataFrame] | None) -> DimensionResult:  # noqa: ARG001
    if not splits:
        return DimensionResult(dimension=QualificationDimensionKind.SAFETY, score=0.0, warnings=("artifacts unreadable -- safety checks skipped",))

    findings: list[str] = []
    warnings: list[str] = []
    for split_name, df in sorted(splits.items()):
        reserved = [c for c in df.columns if any(str(c).startswith(p) for p in _RESERVED_LABEL_PREFIXES) and c != "label"]
        if reserved:
            warnings.append(f"split {split_name!r}: reserved label/target-prefixed column(s) present among feature columns: {reserved}")
        if "label" in df.columns and len(df) > 0:
            non_finite = df["label"].map(lambda v: isinstance(v, float) and (v != v or v in (float("inf"), float("-inf")))).sum()
            if non_finite:
                warnings.append(f"split {split_name!r}: label column contains {int(non_finite)} non-finite value(s)")
    findings.append(f"checked {len(splits)} split(s) for reserved-column leakage and non-finite label values")

    score = _clamp_score(1.0 - 0.25 * len(warnings))
    recommendations = ["Remove or rename any reserved label_/target_-prefixed feature column before using this dataset."] if warnings else []
    return DimensionResult(dimension=QualificationDimensionKind.SAFETY, score=score, findings=tuple(findings), warnings=tuple(warnings), recommendations=tuple(recommendations))
