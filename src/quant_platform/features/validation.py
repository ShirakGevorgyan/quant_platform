"""Structural validation for a built research dataset -- the Milestone 3
analogue of `historical.quality.run_quality_checks`, reusing the same
`Severity` enum and never-raises-itself philosophy: `validate_research_dataset`
always returns a `ResearchDatasetValidationReport`; whether a CRITICAL
finding blocks dataset publication is the caller's policy decision
(`dataset_builder`'s configurable severity gate), not this module's."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from quant_platform.features.missing import missingness_summary
from quant_platform.features.splitting import SplitPlan
from quant_platform.historical.quality import Severity


class DatasetIssueType(Enum):
    DUPLICATE_TIMESTAMP = "DUPLICATE_TIMESTAMP"
    NON_MONOTONIC_TIMESTAMP = "NON_MONOTONIC_TIMESTAMP"
    EXCESSIVE_MISSINGNESS = "EXCESSIVE_MISSINGNESS"
    CONSTANT_FEATURE = "CONSTANT_FEATURE"
    INFINITE_VALUE = "INFINITE_VALUE"
    EXTREME_OUTLIER = "EXTREME_OUTLIER"
    TRAIN_TEST_OVERLAP = "TRAIN_TEST_OVERLAP"
    TARGET_LEAKAGE_SUSPECTED = "TARGET_LEAKAGE_SUSPECTED"
    STALE_EXTERNAL_FEATURE = "STALE_EXTERNAL_FEATURE"
    MISSING_FEATURE_METADATA = "MISSING_FEATURE_METADATA"


@dataclass(frozen=True, slots=True)
class DatasetIssue:
    issue_type: DatasetIssueType
    severity: Severity
    message: str
    affected_columns: tuple[str, ...] = ()
    stats: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ValidationThresholds:
    max_missing_fraction: float = 0.5
    constant_std_epsilon: float = 1e-12
    outlier_zscore_threshold: float = 8.0
    max_stale_fraction: float = 0.5
    target_leakage_correlation_threshold: float = 0.999

    def __post_init__(self) -> None:
        if not (0.0 <= self.max_missing_fraction <= 1.0):
            raise ValueError("max_missing_fraction must be in [0, 1]")
        if self.constant_std_epsilon < 0:
            raise ValueError("constant_std_epsilon must be >= 0")
        if self.outlier_zscore_threshold <= 0:
            raise ValueError("outlier_zscore_threshold must be positive")
        if not (0.0 <= self.max_stale_fraction <= 1.0):
            raise ValueError("max_stale_fraction must be in [0, 1]")
        if not (0.0 < self.target_leakage_correlation_threshold <= 1.0):
            raise ValueError("target_leakage_correlation_threshold must be in (0, 1]")


@dataclass(slots=True)
class ResearchDatasetValidationReport:
    row_count: int
    generated_at: pd.Timestamp
    issues: list[DatasetIssue] = field(default_factory=list)

    @property
    def critical_issues(self) -> list[DatasetIssue]:
        return [i for i in self.issues if i.severity is Severity.CRITICAL]

    @property
    def warnings(self) -> list[DatasetIssue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def infos(self) -> list[DatasetIssue]:
        return [i for i in self.issues if i.severity is Severity.INFO]

    @property
    def is_valid(self) -> bool:
        return not self.critical_issues

    def summary(self) -> str:
        lines = [
            f"ResearchDatasetValidationReport(rows={self.row_count}, valid={self.is_valid}, "
            f"critical={len(self.critical_issues)}, warnings={len(self.warnings)}, info={len(self.infos)})"
        ]
        for issue in self.issues:
            lines.append(f"  [{issue.severity.value}] {issue.issue_type.value}: {issue.message}")
        return "\n".join(lines)


def validate_research_dataset(
    features: pd.DataFrame,
    *,
    timestamps: pd.Series,
    labels: pd.Series | None = None,
    lineage_feature_names: set[str] | None = None,
    split_plan: SplitPlan | None = None,
    thresholds: ValidationThresholds | None = None,
) -> ResearchDatasetValidationReport:
    thresholds = thresholds or ValidationThresholds()
    issues: list[DatasetIssue] = []

    if timestamps.duplicated().any():
        dup_count = int(timestamps.duplicated().sum())
        issues.append(
            DatasetIssue(
                DatasetIssueType.DUPLICATE_TIMESTAMP, Severity.CRITICAL,
                f"{dup_count} duplicate timestamp(s) in the dataset index",
            )
        )
    if len(timestamps) > 1 and not timestamps.is_monotonic_increasing:
        issues.append(
            DatasetIssue(
                DatasetIssueType.NON_MONOTONIC_TIMESTAMP, Severity.CRITICAL,
                "Dataset timestamps are not monotonically increasing",
            )
        )

    summary = missingness_summary(features)
    summary_columns = summary["columns"]
    assert isinstance(summary_columns, dict)
    for col, raw_stats in summary_columns.items():
        assert isinstance(raw_stats, dict)
        null_fraction = float(raw_stats["null_fraction"])
        if null_fraction > thresholds.max_missing_fraction:
            issues.append(
                DatasetIssue(
                    DatasetIssueType.EXCESSIVE_MISSINGNESS, Severity.WARNING,
                    f"Column {col!r} is {null_fraction:.1%} null (threshold "
                    f"{thresholds.max_missing_fraction:.1%})",
                    affected_columns=(col,), stats={"null_fraction": null_fraction},
                )
            )

    numeric_columns = [c for c in features.columns if pd.api.types.is_numeric_dtype(features[c])]
    for col in numeric_columns:
        clean = features[col].dropna()
        if len(clean) == 0:
            continue
        std = float(clean.std())
        if std < thresholds.constant_std_epsilon:
            issues.append(
                DatasetIssue(
                    DatasetIssueType.CONSTANT_FEATURE, Severity.WARNING,
                    f"Column {col!r} is constant (or near-constant) across all non-null rows (std={std:.3g})",
                    affected_columns=(col,), stats={"std": std},
                )
            )
        inf_count = int(np.isinf(clean.to_numpy(dtype="float64")).sum())
        if inf_count > 0:
            issues.append(
                DatasetIssue(
                    DatasetIssueType.INFINITE_VALUE, Severity.CRITICAL,
                    f"Column {col!r} contains {inf_count} infinite value(s)",
                    affected_columns=(col,), stats={"infinite_count": float(inf_count)},
                )
            )
        if std > 0 and len(clean) > 10:
            z = (clean - clean.mean()) / std
            extreme_count = int((z.abs() > thresholds.outlier_zscore_threshold).sum())
            if extreme_count > 0:
                issues.append(
                    DatasetIssue(
                        DatasetIssueType.EXTREME_OUTLIER, Severity.INFO,
                        f"Column {col!r} has {extreme_count} value(s) beyond "
                        f"{thresholds.outlier_zscore_threshold} standard deviations",
                        affected_columns=(col,), stats={"extreme_count": float(extreme_count)},
                    )
                )
        if col.endswith("_is_stale"):
            stale_fraction = float(features[col].astype("float64").mean())
            if stale_fraction > thresholds.max_stale_fraction:
                issues.append(
                    DatasetIssue(
                        DatasetIssueType.STALE_EXTERNAL_FEATURE, Severity.WARNING,
                        f"Column {col!r} is stale {stale_fraction:.1%} of rows (threshold "
                        f"{thresholds.max_stale_fraction:.1%})",
                        affected_columns=(col,), stats={"stale_fraction": stale_fraction},
                    )
                )

    if lineage_feature_names is not None:
        missing_lineage = sorted(set(features.columns) - lineage_feature_names)
        if missing_lineage:
            issues.append(
                DatasetIssue(
                    DatasetIssueType.MISSING_FEATURE_METADATA, Severity.WARNING,
                    f"{len(missing_lineage)} feature column(s) have no recorded lineage: {missing_lineage}",
                    affected_columns=tuple(missing_lineage),
                )
            )

    if labels is not None:
        for col in numeric_columns:
            combined = pd.concat([features[col], labels], axis=1).dropna()
            if len(combined) <= 10:
                continue
            correlation = combined.iloc[:, 0].corr(combined.iloc[:, 1])
            if pd.notna(correlation) and abs(correlation) >= thresholds.target_leakage_correlation_threshold:
                issues.append(
                    DatasetIssue(
                        DatasetIssueType.TARGET_LEAKAGE_SUSPECTED, Severity.CRITICAL,
                        f"Column {col!r} correlates {correlation:.4f} with the label -- suspiciously "
                        "close to a direct copy; verify this feature does not itself encode future/label information",
                        affected_columns=(col,), stats={"correlation": float(correlation)},
                    )
                )

    if split_plan is not None:
        issues.extend(_check_split_overlaps(split_plan))

    return ResearchDatasetValidationReport(row_count=len(features), generated_at=pd.Timestamp.now(tz="UTC"), issues=issues)


def _split_role(name: str) -> str:
    """"train" for a training-partition split ("train", or "fold_k_train"),
    "eval" for anything else ("validation", "test", or "fold_k_test")."""
    return "train" if name == "train" or name.endswith("_train") else "eval"


def _split_group(name: str) -> str:
    """Which fold/plan a split belongs to -- "global" for a chronological
    plan's train/validation/test, "fold_k" for a walk-forward fold's own
    train/test pair. Mirrors `dataset_builder._fold_groups`'s naming
    convention (kept independent, not imported, to avoid a cross-module
    dependency for what is fundamentally just a split-name parsing rule)."""
    if name in ("train", "validation", "test"):
        return "global"
    return "_".join(name.split("_")[:-1]) or name


def _check_split_overlaps(split_plan: SplitPlan) -> list[DatasetIssue]:
    """Flag genuinely suspicious row overlap between splits -- NEVER two
    "eval" splits (that always indicates a real leak, e.g. validation and
    test computed over overlapping ranges, or two walk-forward test folds
    overlapping), and a "train" vs "eval" split only WITHIN the same
    fold/plan (across different walk-forward folds, an earlier fold's test
    rows legitimately becoming part of a LATER fold's training data is the
    expected, correct behavior of expanding/rolling walk-forward CV, not a
    leak -- see `dataset_builder`'s module docstring)."""
    indices_by_split = {s.name: {int(i) for i in s.indices} for s in split_plan.splits}
    names = list(indices_by_split.keys())
    issues: list[DatasetIssue] = []
    for i, name_a in enumerate(names):
        role_a, group_a = _split_role(name_a), _split_group(name_a)
        for name_b in names[i + 1 :]:
            role_b, group_b = _split_role(name_b), _split_group(name_b)
            if role_a == "train" and role_b == "train":
                continue
            if role_a != role_b and group_a != group_b:
                continue
            overlap = indices_by_split[name_a] & indices_by_split[name_b]
            if overlap:
                issues.append(
                    DatasetIssue(
                        DatasetIssueType.TRAIN_TEST_OVERLAP, Severity.CRITICAL,
                        f"Splits {name_a!r} and {name_b!r} share {len(overlap)} row position(s)",
                        affected_columns=(), stats={"overlap_count": float(len(overlap))},
                    )
                )
    return issues


__all__ = [
    "DatasetIssue",
    "DatasetIssueType",
    "ResearchDatasetValidationReport",
    "ValidationThresholds",
    "validate_research_dataset",
]
