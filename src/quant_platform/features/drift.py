"""Descriptive feature stability/drift diagnostics -- research diagnostics,
NOT production monitoring and NOT an automatic feature-selection mechanism.
`compare_splits` only ever reports numbers; nothing in this module drops,
reweights, or otherwise acts on a feature based on its own findings (per
Milestone 3 Section 14's explicit "do not use these diagnostics to select
features automatically yet")."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ColumnSummaryStats:
    count: int
    mean: float
    std: float
    minimum: float
    q25: float
    median: float
    q75: float
    maximum: float
    null_fraction: float


def summarize_column(series: pd.Series) -> ColumnSummaryStats:
    clean = series.dropna()
    total = len(series)
    if len(clean) == 0:
        nan = float("nan")
        return ColumnSummaryStats(0, nan, nan, nan, nan, nan, nan, nan, 1.0 if total else 0.0)
    return ColumnSummaryStats(
        count=len(clean), mean=float(clean.mean()), std=float(clean.std()), minimum=float(clean.min()),
        q25=float(clean.quantile(0.25)), median=float(clean.median()), q75=float(clean.quantile(0.75)),
        maximum=float(clean.max()), null_fraction=float(series.isna().mean()) if total else 0.0,
    )


def population_stability_index(expected: pd.Series, actual: pd.Series, *, bins: int = 10) -> float:
    """Standard PSI: bucket `expected` into `bins` quantile bins, compare
    the same bin edges' population share in `actual`. Returns `NaN` if
    either series has no non-null values, `0.0` if `expected` has no
    variation to bin (every value identical)."""
    expected_clean = expected.dropna().to_numpy(dtype="float64")
    actual_clean = actual.dropna().to_numpy(dtype="float64")
    if len(expected_clean) == 0 or len(actual_clean) == 0:
        return float("nan")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(expected_clean, quantiles))
    if len(edges) < 2:
        return 0.0
    # Extend the outermost edges to +/-inf so a comparison distribution that
    # has drifted entirely outside the reference's observed range still
    # lands in the extreme bins instead of being silently excluded from
    # every bin (which would make `actual_counts.sum()` zero and produce a
    # NaN PSI for exactly the most-drifted case this metric exists to catch).
    edges[0], edges[-1] = -np.inf, np.inf

    expected_counts, _ = np.histogram(expected_clean, bins=edges)
    actual_counts, _ = np.histogram(actual_clean, bins=edges)
    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()

    epsilon = 1e-6
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


@dataclass(frozen=True, slots=True)
class FeatureDriftReport:
    feature_name: str
    reference_stats: ColumnSummaryStats
    comparison_stats: ColumnSummaryStats
    population_stability_index: float
    mean_shift: float
    std_shift: float


@dataclass(frozen=True, slots=True)
class DriftReport:
    reference_name: str
    comparison_name: str
    feature_reports: tuple[FeatureDriftReport, ...]
    highly_correlated_pairs: tuple[tuple[str, str, float], ...]
    constant_features: tuple[str, ...]
    near_constant_features: tuple[str, ...]


def compare_splits(
    reference_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    *,
    reference_name: str = "train",
    comparison_name: str = "comparison",
    correlation_threshold: float = 0.95,
    near_constant_std_threshold: float = 1e-6,
    psi_bins: int = 10,
) -> DriftReport:
    """Compare `comparison_df` (e.g. validation or test features) against
    `reference_df` (e.g. train features): per-column summary stats, PSI,
    mean/std shift, plus reference-only highly-correlated-pair and
    constant/near-constant detection."""
    shared_numeric_columns = [
        c for c in reference_df.columns
        if c in comparison_df.columns and pd.api.types.is_numeric_dtype(reference_df[c])
    ]
    feature_reports = tuple(
        FeatureDriftReport(
            feature_name=col,
            reference_stats=(ref_stats := summarize_column(reference_df[col])),
            comparison_stats=(cmp_stats := summarize_column(comparison_df[col])),
            population_stability_index=population_stability_index(reference_df[col], comparison_df[col], bins=psi_bins),
            mean_shift=cmp_stats.mean - ref_stats.mean,
            std_shift=cmp_stats.std - ref_stats.std,
        )
        for col in shared_numeric_columns
    )

    numeric_columns = [c for c in reference_df.columns if pd.api.types.is_numeric_dtype(reference_df[c])]
    correlation_values = reference_df[numeric_columns].corr().to_numpy(dtype="float64")
    highly_correlated: list[tuple[str, str, float]] = []
    for i, col_a in enumerate(numeric_columns):
        for j, col_b in enumerate(numeric_columns[i + 1 :], start=i + 1):
            value = float(correlation_values[i, j])
            if not np.isnan(value) and abs(value) >= correlation_threshold:
                highly_correlated.append((col_a, col_b, value))

    stds = {c: reference_df[c].std(skipna=True) for c in numeric_columns}
    constant_features = tuple(c for c, std in stds.items() if pd.isna(std) or std == 0)
    near_constant_features = tuple(
        c for c, std in stds.items() if pd.notna(std) and 0 < std < near_constant_std_threshold
    )

    return DriftReport(
        reference_name=reference_name, comparison_name=comparison_name, feature_reports=feature_reports,
        highly_correlated_pairs=tuple(highly_correlated), constant_features=constant_features,
        near_constant_features=near_constant_features,
    )


def render_drift_report(report: DriftReport) -> str:
    lines = [f"Drift report: {report.comparison_name!r} vs. {report.reference_name!r}", "=" * 60]
    for fr in sorted(report.feature_reports, key=lambda r: abs(r.population_stability_index) if pd.notna(r.population_stability_index) else -1, reverse=True):
        lines.append(
            f"{fr.feature_name}: PSI={fr.population_stability_index:.4f} mean_shift={fr.mean_shift:+.4g} "
            f"std_shift={fr.std_shift:+.4g}"
        )
    if report.constant_features:
        lines.append(f"\nConstant features: {list(report.constant_features)}")
    if report.near_constant_features:
        lines.append(f"Near-constant features: {list(report.near_constant_features)}")
    if report.highly_correlated_pairs:
        lines.append("\nHighly correlated pairs:")
        for a, b, corr in report.highly_correlated_pairs:
            lines.append(f"  {a} <-> {b}: {corr:.4f}")
    return "\n".join(lines)


__all__ = [
    "ColumnSummaryStats",
    "DriftReport",
    "FeatureDriftReport",
    "compare_splits",
    "population_stability_index",
    "render_drift_report",
    "summarize_column",
]
