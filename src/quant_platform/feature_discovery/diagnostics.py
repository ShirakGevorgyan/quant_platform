"""`FeatureSignalDiagnostics` (Milestone 11, Phase 2, Part 1): the 10
required-dimension evaluators, one function each, plus
`compute_feature_signal_diagnostics` which runs all 10 for a single
feature and bundles the result. Mirrors `qualification.dimensions`'s
own shape closely (pure functions, one per dimension, sharing
once-computed FACTS rather than re-deriving them) but scoped to a
single FEATURE at a time rather than a whole dataset.

BLOCKING-CODE-TO-DIMENSION MAPPING (the 6 named codes, documented once,
here, rather than scattered):

    LEAKAGE                    -> Leakage Safety
    FUTURE_VISIBILITY          -> Leakage Safety
    AVAILABILITY_VIOLATION     -> Leakage Safety
    NON_DETERMINISTIC_FEATURE  -> Determinism
    IDENTITY_MISMATCH          -> Reproducibility
    MANIFEST_MISMATCH          -> Reproducibility

`AVAILABILITY_VIOLATION` is owned by Leakage Safety, not the
`AVAILABILITY` dimension -- the spec's own "LEAKAGE" section explicitly
lists "availability violations... macro timing... cross asset timing...
dataset timing" as things Leakage Safety itself must verify. The
`AVAILABILITY` dimension is therefore purely descriptive (continuity/
gap statistics), like `COVERAGE` -- neither is ever blocking, mirroring
`qualification.dimensions`'s own "some dimensions never block" pattern.

Reuses (never reimplements): `qualification.verifier`'s pure
`verify_identity`/`verify_artifacts`/`verify_lineage`/
`verify_no_future_leakage` functions (NOT `QualificationVerifier`/
`DatasetQualificationEngine` themselves -- this package never
instantiates either), `features.validation.validate_research_dataset`,
`features.drift.compare_splits`/`population_stability_index`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.feature_discovery.evidence import (
    FEATURE_DISCOVERY_DIMENSION_ORDER,
    BlockingFindingCode,
    FeatureDiscoveryDimensionKind,
    make_evidence,
)
from quant_platform.feature_discovery.models import FeatureDimensionResult, FeatureSignalDiagnostics
from quant_platform.feature_discovery.statistics import (
    FeatureStatistics,
    compute_feature_statistics,
    rolling_window_size,
)
from quant_platform.features.drift import compare_splits, population_stability_index
from quant_platform.features.manifests import ResearchDatasetManifest, ResearchDatasetStore
from quant_platform.features.validation import (
    DatasetIssueType,
    ResearchDatasetValidationReport,
    validate_research_dataset,
)
from quant_platform.historical.quality import Severity
from quant_platform.qualification.verifier import (
    verify_artifacts,
    verify_identity,
    verify_lineage,
    verify_no_future_leakage,
)

__all__ = [
    "SharedDiscoveryFacts",
    "compute_feature_signal_diagnostics",
    "compute_shared_discovery_facts",
    "evaluate_availability",
    "evaluate_coverage",
    "evaluate_determinism",
    "evaluate_drift_behaviour",
    "evaluate_information_content",
    "evaluate_leakage_safety",
    "evaluate_redundancy",
    "evaluate_regime_stability",
    "evaluate_reproducibility",
    "evaluate_temporal_stability",
]

_PSI_WARNING_THRESHOLD = 0.25
_NEAR_REDUNDANCY_THRESHOLD = 0.95
_PERFECT_REDUNDANCY_THRESHOLD = 0.999
_MISSING_WINDOW_MIN_RUN = 5


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _finite_series(series: pd.Series) -> pd.Series:
    """`+/-inf` poisons `features.drift.population_stability_index`'s own quantile/histogram
    arithmetic (an M3 primitive, reused here unmodified rather than patched) with `inf - inf = NaN`
    warnings -- callers that feed a series into PSI/rolling computations mask infinities to NaN
    first, matching `statistics.py`'s own `finite_series` convention."""
    return series.mask(np.isinf(series), other=np.nan)


def _null_runs(is_null: pd.Series) -> list[tuple[int, int]]:
    """Every maximal run of consecutive `True` values in `is_null`, as
    `(start_index, end_index_exclusive)` pairs, in row-position order."""
    runs: list[tuple[int, int]] = []
    values = is_null.to_numpy()
    start: int | None = None
    for i, flag in enumerate(values):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


# --------------------------------------------------------------------------
# Shared, once-per-dataset facts every dimension evaluator consumes.
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SharedDiscoveryFacts:
    manifest: ResearchDatasetManifest
    splits: dict[str, pd.DataFrame] | None
    artifacts_readable: bool
    artifacts_error_message: str | None
    metadata_checksums_match_manifest: bool
    row_counts_match_manifest: bool
    identity_matches: bool
    identity_message: str | None
    lineage_present: bool
    missing_lineage_fields: tuple[str, ...]
    validation_report: ResearchDatasetValidationReport | None
    """`validate_research_dataset` over the train split, with labels --
    the single, shared source of per-feature `TARGET_LEAKAGE_SUSPECTED`
    findings (filtered by `affected_columns` per feature)."""
    redundancy_constant_features: tuple[str, ...]
    redundancy_near_constant_features: tuple[str, ...]
    redundancy_correlated_pairs: tuple[tuple[str, str, float], ...]
    redundancy_exact_duplicate_pairs: tuple[tuple[str, str], ...]


def compute_shared_discovery_facts(manifest: ResearchDatasetManifest, research_store: ResearchDatasetStore) -> SharedDiscoveryFacts:
    identity_matches, identity_message = verify_identity(manifest)
    artifacts = verify_artifacts(research_store, manifest)
    lineage_present, missing_lineage_fields = verify_lineage(manifest)

    validation_report: ResearchDatasetValidationReport | None = None
    constant_features: tuple[str, ...] = ()
    near_constant_features: tuple[str, ...] = ()
    correlated_pairs: tuple[tuple[str, str, float], ...] = ()
    exact_duplicate_pairs: tuple[tuple[str, str], ...] = ()

    if artifacts.readable and artifacts.splits and "train" in artifacts.splits and len(artifacts.splits["train"]):
        train_df = artifacts.splits["train"]
        feature_columns = [c for c in manifest.feature_names if c in train_df.columns]
        if feature_columns and "label" in train_df.columns:
            validation_report = validate_research_dataset(
                train_df[feature_columns], timestamps=train_df["open_time"] if "open_time" in train_df.columns else pd.Series(range(len(train_df))),
                labels=train_df["label"],
            )
        numeric_columns = [c for c in feature_columns if pd.api.types.is_numeric_dtype(train_df[c])]
        if numeric_columns:
            finite_numeric = train_df[numeric_columns].mask(np.isinf(train_df[numeric_columns]), other=np.nan)
            drift = compare_splits(finite_numeric, finite_numeric, reference_name="train", comparison_name="train")
            constant_features = drift.constant_features
            near_constant_features = drift.near_constant_features
            correlated_pairs = drift.highly_correlated_pairs
            duplicates: list[tuple[str, str]] = []
            for i, col_a in enumerate(numeric_columns):
                for col_b in numeric_columns[i + 1 :]:
                    if train_df[col_a].equals(train_df[col_b]):
                        duplicates.append((col_a, col_b))
            exact_duplicate_pairs = tuple(duplicates)

    return SharedDiscoveryFacts(
        manifest=manifest, splits=artifacts.splits, artifacts_readable=artifacts.readable, artifacts_error_message=artifacts.error_message,
        metadata_checksums_match_manifest=artifacts.metadata_checksums_match_manifest, row_counts_match_manifest=artifacts.row_counts_match_manifest,
        identity_matches=identity_matches, identity_message=identity_message, lineage_present=lineage_present,
        missing_lineage_fields=missing_lineage_fields, validation_report=validation_report,
        redundancy_constant_features=constant_features, redundancy_near_constant_features=near_constant_features,
        redundancy_correlated_pairs=correlated_pairs, redundancy_exact_duplicate_pairs=exact_duplicate_pairs,
    )


# --------------------------------------------------------------------------
# 1. Information Content -- never blocking
# --------------------------------------------------------------------------
def evaluate_information_content(feature_name: str, stats: FeatureStatistics) -> FeatureDimensionResult:
    dim = FeatureDiscoveryDimensionKind.INFORMATION_CONTENT
    evidence = [make_evidence(
        finding=f"variance={stats.variance:.6g} entropy={stats.entropy:.4f} cardinality={stats.cardinality} unique_ratio={stats.unique_ratio:.4f}",
        evidence=(f"mean={stats.mean:.6g}", f"std={stats.std:.6g}", f"effective_cardinality={stats.effective_cardinality:.4f}"),
        dimension=dim, severity=Severity.INFO, affected_feature=feature_name,
        supporting_statistics={"variance": stats.variance, "entropy": stats.entropy, "unique_ratio": stats.unique_ratio},
    )]
    penalty = 0.0
    if stats.constant_ratio == 1.0:
        evidence.append(make_evidence(
            finding="feature is constant (zero variance)", evidence=(f"cardinality={stats.cardinality}",), dimension=dim,
            severity=Severity.WARNING, affected_feature=feature_name, supporting_statistics={"constant_ratio": stats.constant_ratio},
            recommendation="A constant feature carries no information for any downstream model; verify this is expected.",
        ))
        penalty += 0.5
    elif stats.near_constant_ratio > 0.95:
        evidence.append(make_evidence(
            finding=f"feature is near-constant ({stats.near_constant_ratio:.1%} of values share the mode)", evidence=(),
            dimension=dim, severity=Severity.WARNING, affected_feature=feature_name,
            supporting_statistics={"near_constant_ratio": stats.near_constant_ratio},
        ))
        penalty += 0.25
    if stats.missing_ratio > 0.5:
        evidence.append(make_evidence(
            finding=f"missing_ratio={stats.missing_ratio:.1%} exceeds 50%", evidence=(), dimension=dim, severity=Severity.WARNING,
            affected_feature=feature_name, supporting_statistics={"missing_ratio": stats.missing_ratio},
        ))
        penalty += 0.2
    if stats.infinite_ratio > 0.0:
        evidence.append(make_evidence(
            finding=f"non-finite (+/-inf) values present: infinite_ratio={stats.infinite_ratio:.4%}", evidence=(), dimension=dim,
            severity=Severity.CRITICAL, affected_feature=feature_name, supporting_statistics={"infinite_ratio": stats.infinite_ratio},
            recommendation="Trace the feature computation that produced an infinite value -- likely a division by zero. All other statistics for this feature exclude infinite values.",
        ))
        penalty += 0.3
    score = _clamp_score(1.0 - penalty)
    return FeatureDimensionResult(dimension=dim, feature_name=feature_name, score=score, evidence=tuple(evidence))


# --------------------------------------------------------------------------
# 2. Temporal Stability -- never blocking
# --------------------------------------------------------------------------
def evaluate_temporal_stability(feature_name: str, series: pd.Series, stats: FeatureStatistics, *, n_windows: int = 5) -> FeatureDimensionResult:
    dim = FeatureDiscoveryDimensionKind.TEMPORAL_STABILITY
    evidence = []
    penalty = 0.0

    series = _finite_series(series)
    clean = series.dropna()
    if len(clean) >= n_windows * 5:
        chunks = np.array_split(clean.to_numpy(dtype="float64"), n_windows)
        chunk_means = np.array([c.mean() for c in chunks if len(c)])
        mean_cv = float(chunk_means.std() / abs(chunk_means.mean())) if chunk_means.mean() != 0 else 0.0
        evidence.append(make_evidence(
            finding=f"window consistency: {n_windows}-chunk mean coefficient of variation={mean_cv:.4f}", evidence=(f"chunk_means={chunk_means.tolist()}",),
            dimension=dim, severity=(Severity.WARNING if mean_cv > 1.0 else Severity.INFO), affected_feature=feature_name,
            supporting_statistics={"window_mean_cv": mean_cv},
        ))
        if mean_cv > 1.0:
            penalty += 0.15
        expanding_mean = series.expanding(min_periods=max(5, len(series) // 10)).mean().dropna()
        expanding_cv = float(expanding_mean.std() / abs(expanding_mean.mean())) if len(expanding_mean) and expanding_mean.mean() != 0 else 0.0
        evidence.append(make_evidence(
            finding=f"expanding consistency: expanding-mean coefficient of variation={expanding_cv:.4f}", evidence=(),
            dimension=dim, severity=(Severity.WARNING if expanding_cv > 1.0 else Severity.INFO), affected_feature=feature_name,
            supporting_statistics={"expanding_mean_cv": expanding_cv},
        ))
        if expanding_cv > 1.0:
            penalty += 0.1

    evidence.append(make_evidence(
        finding=f"rolling consistency: rolling-variance CV={stats.rolling_variance_cv:.4f} rolling-entropy CV={stats.rolling_entropy_cv:.4f}",
        evidence=(), dimension=dim, severity=(Severity.WARNING if stats.rolling_variance_cv > 1.5 else Severity.INFO),
        affected_feature=feature_name, supporting_statistics={"rolling_variance_cv": stats.rolling_variance_cv, "rolling_entropy_cv": stats.rolling_entropy_cv},
    ))
    if stats.rolling_variance_cv > 1.5:
        penalty += 0.15

    evidence.append(make_evidence(
        finding=f"availability persistence: rolling missingness mean={stats.rolling_missingness_mean:.4f} max_swing={stats.rolling_missingness_max_swing:.4f}",
        evidence=(), dimension=dim, severity=(Severity.WARNING if stats.rolling_missingness_max_swing > 0.3 else Severity.INFO),
        affected_feature=feature_name, supporting_statistics={"rolling_missingness_max_swing": stats.rolling_missingness_max_swing},
        recommendation=("This feature's availability changes materially over time -- confirm this is expected (e.g. a source that came online partway through)." if stats.rolling_missingness_max_swing > 0.3 else None),
    ))
    if stats.rolling_missingness_max_swing > 0.3:
        penalty += 0.2

    evidence.append(make_evidence(
        finding=f"feature persistence: missing_ratio={stats.missing_ratio:.4f}", evidence=(), dimension=dim, severity=Severity.INFO,
        affected_feature=feature_name, supporting_statistics={"missing_ratio": stats.missing_ratio},
    ))

    score = _clamp_score(1.0 - penalty)
    return FeatureDimensionResult(dimension=dim, feature_name=feature_name, score=score, evidence=tuple(evidence))


# --------------------------------------------------------------------------
# 3. Regime Stability -- never blocking. Classifies STABILITY only, never alpha.
# --------------------------------------------------------------------------
def evaluate_regime_stability(feature_name: str, series: pd.Series) -> FeatureDimensionResult:
    dim = FeatureDiscoveryDimensionKind.REGIME_STABILITY
    evidence = []
    penalty = 0.0

    series = _finite_series(series)
    clean = series.dropna()
    window = rolling_window_size(len(series))
    if len(clean) >= window * 4:
        rolling_std = series.rolling(window=window, min_periods=window).std()
        median_vol = rolling_std.median()
        if pd.notna(median_vol) and median_vol > 0:
            high_vol_mask = (rolling_std > median_vol) & series.notna()
            low_vol_mask = (rolling_std <= median_vol) & series.notna()
            if high_vol_mask.sum() >= 10 and low_vol_mask.sum() >= 10:
                psi = population_stability_index(series[low_vol_mask], series[high_vol_mask])
                evidence.append(make_evidence(
                    finding=f"volatility regime: PSI(low-vol vs high-vol)={psi:.4f}", evidence=(f"low_vol_rows={int(low_vol_mask.sum())}", f"high_vol_rows={int(high_vol_mask.sum())}"),
                    dimension=dim, severity=(Severity.WARNING if psi == psi and psi >= _PSI_WARNING_THRESHOLD else Severity.INFO),
                    affected_feature=feature_name, supporting_statistics={"volatility_regime_psi": float(psi) if psi == psi else 0.0},
                    recommendation=("This feature's own distribution shifts substantially between its own high- and low-volatility sub-periods." if psi == psi and psi >= _PSI_WARNING_THRESHOLD else None),
                ))
                if psi == psi and psi >= _PSI_WARNING_THRESHOLD:
                    penalty += 0.2

        rolling_mean = series.rolling(window=window, min_periods=window).mean()
        trend = rolling_mean.diff()
        trend_threshold = 0.1 * trend.std() if pd.notna(trend.std()) and trend.std() > 0 else 0.0
        if trend_threshold > 0:
            bull_mask = (trend > trend_threshold) & series.notna()
            bear_mask = (trend < -trend_threshold) & series.notna()
            pairwise_psi = []
            if bull_mask.sum() >= 10 and bear_mask.sum() >= 10:
                pairwise_psi.append(population_stability_index(series[bull_mask], series[bear_mask]))
            if pairwise_psi:
                max_psi = max(p for p in pairwise_psi if p == p) if any(p == p for p in pairwise_psi) else float("nan")
                evidence.append(make_evidence(
                    finding=f"trend regime: max PSI across bull/bear sub-periods={max_psi:.4f}" if max_psi == max_psi else "trend regime: insufficient bull/bear rows to compare",
                    evidence=(f"bull_rows={int(bull_mask.sum())}", f"bear_rows={int(bear_mask.sum())}"), dimension=dim,
                    severity=(Severity.WARNING if max_psi == max_psi and max_psi >= _PSI_WARNING_THRESHOLD else Severity.INFO),
                    affected_feature=feature_name, supporting_statistics={"trend_regime_psi": float(max_psi) if max_psi == max_psi else 0.0},
                ))
                if max_psi == max_psi and max_psi >= _PSI_WARNING_THRESHOLD:
                    penalty += 0.2

    evidence.append(make_evidence(
        finding="macro tightening/easing regime: out of scope for this package (no raw macro source access at feature granularity)",
        evidence=("see market_data_lineage for the closest available build-time macro/cross-asset signal",), dimension=dim,
        severity=Severity.INFO, affected_feature=feature_name,
    ))

    score = _clamp_score(1.0 - penalty)
    return FeatureDimensionResult(dimension=dim, feature_name=feature_name, score=score, evidence=tuple(evidence))


# --------------------------------------------------------------------------
# 4. Drift Behaviour -- never blocking
# --------------------------------------------------------------------------
def evaluate_drift_behaviour(feature_name: str, splits: dict[str, pd.DataFrame] | None) -> FeatureDimensionResult:
    dim = FeatureDiscoveryDimensionKind.DRIFT_BEHAVIOUR
    evidence = []
    penalty = 0.0
    if not splits or "train" not in splits or feature_name not in splits["train"].columns:
        return FeatureDimensionResult(dimension=dim, feature_name=feature_name, score=0.0, evidence=(make_evidence(
            finding="drift checks skipped -- feature not present in a readable train split", evidence=(), dimension=dim,
            severity=Severity.WARNING, affected_feature=feature_name,
        ),))

    train_series = _finite_series(splits["train"][feature_name])
    for comparison_name in ("validation", "test"):
        if comparison_name not in splits or feature_name not in splits[comparison_name].columns or len(splits[comparison_name]) == 0:
            continue
        comparison_series = _finite_series(splits[comparison_name][feature_name])
        psi = population_stability_index(train_series, comparison_series)
        train_mean, train_std = float(train_series.mean()), float(train_series.std())
        comparison_mean, comparison_std = float(comparison_series.mean()), float(comparison_series.std())
        mean_drift = abs(comparison_mean - train_mean) / abs(train_mean) if train_mean != 0 else abs(comparison_mean - train_mean)
        variance_drift = abs(comparison_std - train_std) / train_std if train_std != 0 else abs(comparison_std - train_std)
        evidence.append(make_evidence(
            finding=f"distribution drift: train vs {comparison_name} PSI={psi:.4f} mean_drift={mean_drift:.4f} variance_drift={variance_drift:.4f}",
            evidence=(), dimension=dim, severity=(Severity.WARNING if psi == psi and psi >= _PSI_WARNING_THRESHOLD else Severity.INFO),
            affected_feature=feature_name, supporting_statistics={"psi": float(psi) if psi == psi else 0.0, "mean_drift": mean_drift, "variance_drift": variance_drift},
        ))
        if psi == psi and psi >= _PSI_WARNING_THRESHOLD:
            penalty += 0.15

    window = rolling_window_size(len(train_series))
    clean = train_series.dropna()
    if len(clean) >= window * 3:
        chunks = [clean.iloc[i : i + window] for i in range(0, len(clean), window) if len(clean.iloc[i : i + window]) == window]
        if len(chunks) >= 2:
            reference = chunks[0]
            chunk_psis = [population_stability_index(reference, chunk) for chunk in chunks[1:]]
            valid_psis = [p for p in chunk_psis if p == p]
            max_rolling_psi = max(valid_psis) if valid_psis else 0.0
            evidence.append(make_evidence(
                finding=f"rolling drift: max PSI vs the feature's own first window across {len(chunks) - 1} later window(s)={max_rolling_psi:.4f}",
                evidence=(), dimension=dim, severity=(Severity.WARNING if max_rolling_psi >= _PSI_WARNING_THRESHOLD else Severity.INFO),
                affected_feature=feature_name, supporting_statistics={"max_rolling_psi": max_rolling_psi},
            ))
            if max_rolling_psi >= _PSI_WARNING_THRESHOLD:
                penalty += 0.15

    if not evidence:
        evidence.append(make_evidence(
            finding="drift checks: only one non-empty split available -- no comparison possible", evidence=(), dimension=dim,
            severity=Severity.INFO, affected_feature=feature_name,
        ))
    score = _clamp_score(1.0 - penalty)
    return FeatureDimensionResult(dimension=dim, feature_name=feature_name, score=score, evidence=tuple(evidence))


# --------------------------------------------------------------------------
# 5. Redundancy -- never blocking, report-only (never removes anything)
# --------------------------------------------------------------------------
def evaluate_redundancy(feature_name: str, facts: SharedDiscoveryFacts) -> FeatureDimensionResult:
    dim = FeatureDiscoveryDimensionKind.REDUNDANCY
    evidence = []
    penalty = 0.0

    exact_partners = sorted({b if a == feature_name else a for a, b in facts.redundancy_exact_duplicate_pairs if feature_name in (a, b)})
    if exact_partners:
        evidence.append(make_evidence(
            finding=f"identical/duplicated feature(s) detected: {exact_partners}", evidence=("exact value equality across every row of the train split",),
            dimension=dim, severity=Severity.WARNING, affected_feature=feature_name, supporting_statistics={"exact_duplicate_count": float(len(exact_partners))},
            recommendation="Two exactly-identical feature columns carry no additional information beyond the first; report only, this package does not remove either.",
        ))
        penalty += 0.4

    perfect_partners: list[tuple[str, float]] = []
    near_partners: list[tuple[str, float]] = []
    for a, b, corr in facts.redundancy_correlated_pairs:
        if feature_name not in (a, b):
            continue
        other = b if a == feature_name else a
        if (a, b) in facts.redundancy_exact_duplicate_pairs or (b, a) in facts.redundancy_exact_duplicate_pairs:
            continue  # already reported as an exact duplicate above
        if abs(corr) >= _PERFECT_REDUNDANCY_THRESHOLD:
            perfect_partners.append((other, corr))
        elif abs(corr) >= _NEAR_REDUNDANCY_THRESHOLD:
            near_partners.append((other, corr))

    if perfect_partners:
        evidence.append(make_evidence(
            finding=f"derived duplicate / perfect redundancy: correlated >= {_PERFECT_REDUNDANCY_THRESHOLD} with {[p[0] for p in perfect_partners]}",
            evidence=(f"correlations={perfect_partners}",), dimension=dim, severity=Severity.WARNING, affected_feature=feature_name,
            supporting_statistics={"perfect_redundancy_count": float(len(perfect_partners))},
        ))
        penalty += 0.25
    if near_partners:
        evidence.append(make_evidence(
            finding=f"near redundancy: correlated >= {_NEAR_REDUNDANCY_THRESHOLD} with {[p[0] for p in near_partners]}", evidence=(f"correlations={near_partners}",),
            dimension=dim, severity=Severity.INFO, affected_feature=feature_name, supporting_statistics={"near_redundancy_count": float(len(near_partners))},
        ))
        penalty += 0.1

    if not evidence:
        evidence.append(make_evidence(
            finding="no identical, duplicated, or (near-)redundant feature detected", evidence=(), dimension=dim, severity=Severity.INFO,
            affected_feature=feature_name,
        ))
    score = _clamp_score(1.0 - penalty)
    return FeatureDimensionResult(dimension=dim, feature_name=feature_name, score=score, evidence=tuple(evidence))


# --------------------------------------------------------------------------
# 6. Coverage -- never blocking
# --------------------------------------------------------------------------
def evaluate_coverage(feature_name: str, series: pd.Series, label: pd.Series | None, stats: FeatureStatistics) -> FeatureDimensionResult:
    dim = FeatureDiscoveryDimensionKind.COVERAGE
    evidence = []
    penalty = 0.0

    coverage_fraction = 1.0 - stats.missing_ratio
    evidence.append(make_evidence(
        finding=f"coverage: {coverage_fraction:.1%} of rows have a non-null value", evidence=(), dimension=dim,
        severity=(Severity.WARNING if coverage_fraction < 0.5 else Severity.INFO), affected_feature=feature_name,
        supporting_statistics={"coverage_fraction": coverage_fraction},
    ))
    if coverage_fraction < 0.5:
        penalty += 0.3

    is_null = series.isna()
    runs = _null_runs(is_null)
    warmup_rows = runs[0][1] - runs[0][0] if runs and runs[0][0] == 0 else 0
    evidence.append(make_evidence(
        finding=f"warmup: leading run of {warmup_rows} null row(s)", evidence=(), dimension=dim, severity=Severity.INFO,
        affected_feature=feature_name, supporting_statistics={"warmup_rows": float(warmup_rows)},
    ))

    missing_windows = [r for r in runs if (r[1] - r[0]) >= _MISSING_WINDOW_MIN_RUN and r != (runs[0] if runs and runs[0][0] == 0 else (-1, -1))]
    if missing_windows:
        evidence.append(make_evidence(
            finding=f"missing windows: {len(missing_windows)} non-leading null run(s) of length >= {_MISSING_WINDOW_MIN_RUN}",
            evidence=(f"runs={missing_windows[:10]}",), dimension=dim, severity=Severity.WARNING, affected_feature=feature_name,
            supporting_statistics={"missing_window_count": float(len(missing_windows))},
        ))
        penalty += 0.15

    if label is not None:
        usable_rows = int((series.notna() & label.notna()).sum())
        evidence.append(make_evidence(
            finding=f"usable rows: {usable_rows} row(s) have both a non-null feature value and a non-null label", evidence=(),
            dimension=dim, severity=Severity.INFO, affected_feature=feature_name, supporting_statistics={"usable_rows": float(usable_rows)},
        ))

    score = _clamp_score(1.0 - penalty)
    return FeatureDimensionResult(dimension=dim, feature_name=feature_name, score=score, evidence=tuple(evidence))


# --------------------------------------------------------------------------
# 7. Availability -- never blocking (violations are Leakage Safety's job)
# --------------------------------------------------------------------------
def evaluate_availability(feature_name: str, series: pd.Series, manifest: ResearchDatasetManifest) -> FeatureDimensionResult:
    dim = FeatureDiscoveryDimensionKind.AVAILABILITY
    evidence = []
    penalty = 0.0

    is_null = series.isna()
    runs = _null_runs(is_null)
    longest_gap = max((r[1] - r[0] for r in runs), default=0)
    longest_gap_fraction = longest_gap / len(series) if len(series) else 0.0
    evidence.append(make_evidence(
        finding=f"availability continuity: longest dark period={longest_gap} row(s) ({longest_gap_fraction:.1%} of the split)", evidence=(),
        dimension=dim, severity=(Severity.WARNING if longest_gap_fraction > 0.2 else Severity.INFO), affected_feature=feature_name,
        supporting_statistics={"longest_gap_fraction": longest_gap_fraction},
    ))
    if longest_gap_fraction > 0.2:
        penalty += 0.2

    if manifest.market_data_lineage:
        coverage_decision = manifest.market_data_lineage.get("coverage_decision")
        findings = coverage_decision.get("findings") if isinstance(coverage_decision, dict) else None
        non_ok = [f for f in findings if isinstance(f, dict) and f.get("status") != "ok"] if isinstance(findings, list) else []
        if non_ok:
            evidence.append(make_evidence(
                finding=f"{len(non_ok)} bound macro/cross-asset source(s) have non-ok build-time coverage status -- may affect this feature if it is sourced from one of them",
                evidence=(f"sources={[f.get('source_name') for f in non_ok]}",), dimension=dim, severity=Severity.WARNING,
                affected_feature=feature_name, supporting_statistics={"non_ok_source_count": float(len(non_ok))},
            ))
            penalty += 0.1
    else:
        evidence.append(make_evidence(
            finding="availability: this dataset's base data was not sourced through features.market_data_bridge", evidence=(),
            dimension=dim, severity=Severity.INFO, affected_feature=feature_name,
        ))

    score = _clamp_score(1.0 - penalty)
    return FeatureDimensionResult(dimension=dim, feature_name=feature_name, score=score, evidence=tuple(evidence))


# --------------------------------------------------------------------------
# 8. Leakage Safety -- LEAKAGE, FUTURE_VISIBILITY, AVAILABILITY_VIOLATION
# --------------------------------------------------------------------------
def evaluate_leakage_safety(feature_name: str, facts: SharedDiscoveryFacts, series: pd.Series | None) -> FeatureDimensionResult:
    dim = FeatureDiscoveryDimensionKind.LEAKAGE_SAFETY
    evidence = []
    blocking = False

    if facts.validation_report is not None:
        leakage_issues = [i for i in facts.validation_report.issues if i.issue_type is DatasetIssueType.TARGET_LEAKAGE_SUSPECTED and feature_name in i.affected_columns]
        if leakage_issues:
            for issue in leakage_issues:
                evidence.append(make_evidence(
                    finding=f"leakage: {issue.message}", evidence=(), dimension=dim, severity=Severity.CRITICAL, affected_feature=feature_name,
                    supporting_statistics=issue.stats, blocking=True, blocking_code=BlockingFindingCode.LEAKAGE,
                    recommendation="Investigate this feature for direct or near-direct encoding of the label before any further use.",
                ))
            blocking = True
        else:
            evidence.append(make_evidence(
                finding="leakage: no suspected target leakage found", evidence=(), dimension=dim, severity=Severity.INFO, affected_feature=feature_name,
            ))

    if facts.splits and "train" in facts.splits:
        train_df = facts.splits["train"]
        numeric_columns = train_df.select_dtypes(include="number").columns
        finite_train_df = train_df.copy()
        finite_train_df[numeric_columns] = finite_train_df[numeric_columns].mask(np.isinf(finite_train_df[numeric_columns]), other=np.nan)
        leakage_free, leakage_messages = verify_no_future_leakage(finite_train_df, split_name="train")
        if not leakage_free:
            evidence.append(make_evidence(
                finding="future visibility: temporal-ordering-based leakage check failed for the train split", evidence=leakage_messages,
                dimension=dim, severity=Severity.CRITICAL, affected_feature=feature_name, blocking=True, blocking_code=BlockingFindingCode.FUTURE_VISIBILITY,
            ))
            blocking = True
        else:
            evidence.append(make_evidence(
                finding="future visibility: no temporal-ordering-based leakage found", evidence=(), dimension=dim, severity=Severity.INFO,
                affected_feature=feature_name,
            ))

    if series is not None and "open_time" in (facts.splits or {}).get("train", pd.DataFrame()).columns:
        train_df = facts.splits["train"]  # type: ignore[index]
        non_null_times = train_df.loc[series.notna(), "open_time"]
        out_of_range = non_null_times[(non_null_times < facts.manifest.utc_start) | (non_null_times >= facts.manifest.utc_end)]
        if len(out_of_range):
            evidence.append(make_evidence(
                finding=f"availability violation: {len(out_of_range)} non-null observation(s) fall outside the manifest's declared [utc_start, utc_end) range",
                evidence=(f"first_violation={out_of_range.iloc[0]}",), dimension=dim, severity=Severity.CRITICAL, affected_feature=feature_name,
                blocking=True, blocking_code=BlockingFindingCode.AVAILABILITY_VIOLATION,
                recommendation="A feature value observed outside the dataset's own declared date range indicates a dataset-timing defect; investigate before use.",
            ))
            blocking = True

    macro_findings = []
    if facts.manifest.market_data_lineage:
        coverage_decision = facts.manifest.market_data_lineage.get("coverage_decision")
        raw_findings = coverage_decision.get("findings") if isinstance(coverage_decision, dict) else None
        if isinstance(raw_findings, list):
            macro_findings = [f for f in raw_findings if isinstance(f, dict) and f.get("source_kind") in ("macro", "cross_asset") and f.get("status") != "ok"]
    evidence.append(make_evidence(
        finding=(f"macro/cross-asset timing: {len(macro_findings)} bound source(s) show non-ok build-time coverage status" if macro_findings else "macro/cross-asset timing: no bound source shows a build-time coverage violation"),
        evidence=(), dimension=dim, severity=(Severity.WARNING if macro_findings else Severity.INFO), affected_feature=feature_name,
    ))

    score = 0.0 if blocking else 1.0
    return FeatureDimensionResult(dimension=dim, feature_name=feature_name, score=score, evidence=tuple(evidence))


# --------------------------------------------------------------------------
# 9. Determinism -- NON_DETERMINISTIC_FEATURE
# --------------------------------------------------------------------------
def evaluate_determinism(feature_name: str, facts: SharedDiscoveryFacts) -> FeatureDimensionResult:
    dim = FeatureDiscoveryDimensionKind.DETERMINISM
    if not facts.artifacts_readable:
        return FeatureDimensionResult(dimension=dim, feature_name=feature_name, score=0.0, evidence=(make_evidence(
            finding="determinism checks skipped -- artifacts unreadable", evidence=(f"error={facts.artifacts_error_message}",), dimension=dim,
            severity=Severity.WARNING, affected_feature=feature_name,
        ),))
    if not facts.metadata_checksums_match_manifest or not facts.row_counts_match_manifest:
        evidence = (make_evidence(
            finding="non-deterministic feature: manifest.output_content_hashes/row_counts do not match the content store's own independently-recorded metadata.json",
            evidence=(), dimension=dim, severity=Severity.CRITICAL, affected_feature=feature_name, blocking=True,
            blocking_code=BlockingFindingCode.NON_DETERMINISTIC_FEATURE,
            recommendation="Treat every feature in this dataset version as untrustworthy until the two persisted records are reconciled.",
        ),)
        return FeatureDimensionResult(dimension=dim, feature_name=feature_name, score=0.0, evidence=evidence)
    return FeatureDimensionResult(dimension=dim, feature_name=feature_name, score=1.0, evidence=(make_evidence(
        finding="determinism: manifest and content-store records agree", evidence=(), dimension=dim, severity=Severity.INFO, affected_feature=feature_name,
    ),))


# --------------------------------------------------------------------------
# 10. Reproducibility -- IDENTITY_MISMATCH, MANIFEST_MISMATCH
# --------------------------------------------------------------------------
def evaluate_reproducibility(feature_name: str, facts: SharedDiscoveryFacts) -> FeatureDimensionResult:
    dim = FeatureDiscoveryDimensionKind.REPRODUCIBILITY
    evidence = []
    blocking = False

    if not facts.identity_matches:
        evidence.append(make_evidence(
            finding=facts.identity_message or "dataset_id does not match its own recomputed identity", evidence=(), dimension=dim,
            severity=Severity.CRITICAL, affected_feature=feature_name, blocking=True, blocking_code=BlockingFindingCode.IDENTITY_MISMATCH,
        ))
        blocking = True
    else:
        evidence.append(make_evidence(
            finding="identity: dataset_id is reproducible from the manifest's own recipe fields", evidence=(), dimension=dim,
            severity=Severity.INFO, affected_feature=feature_name,
        ))

    declared = feature_name in facts.manifest.feature_names
    if not facts.artifacts_readable:
        # Artifacts entirely unreadable/nonexistent is itself a manifest/reality
        # mismatch -- the manifest promises artifacts that cannot be read at all.
        # Never silently score this 0.0 without a blocking flag anywhere (unlike
        # Determinism's own "can't even check" branch, which is fine to leave
        # non-blocking here precisely BECAUSE this dimension covers it).
        evidence.append(make_evidence(
            finding=f"manifest mismatch: artifacts unreadable, cannot confirm feature {feature_name!r} is present", evidence=(f"error={facts.artifacts_error_message}",),
            dimension=dim, severity=Severity.CRITICAL, affected_feature=feature_name, blocking=True, blocking_code=BlockingFindingCode.MANIFEST_MISMATCH,
        ))
        blocking = True
    else:
        present_in_splits = bool(facts.splits) and all(feature_name in df.columns for df in (facts.splits or {}).values() if len(df))
        if not declared or not present_in_splits:
            evidence.append(make_evidence(
                finding=f"manifest mismatch: feature {feature_name!r} is declared={declared}, present_in_every_split={present_in_splits}",
                evidence=(), dimension=dim, severity=Severity.CRITICAL, affected_feature=feature_name, blocking=True,
                blocking_code=BlockingFindingCode.MANIFEST_MISMATCH,
            ))
            blocking = True
        else:
            evidence.append(make_evidence(
                finding="manifest: feature is declared and present in every readable split", evidence=(), dimension=dim, severity=Severity.INFO,
                affected_feature=feature_name,
            ))

    score = 0.0 if blocking else 1.0
    return FeatureDimensionResult(dimension=dim, feature_name=feature_name, score=score, evidence=tuple(evidence))


def compute_feature_signal_diagnostics(feature_name: str, facts: SharedDiscoveryFacts) -> FeatureSignalDiagnostics:
    train_series = (facts.splits or {}).get("train", pd.DataFrame()).get(feature_name)
    if train_series is None:
        train_series = pd.Series([], dtype="float64")
    label_series = (facts.splits or {}).get("train", pd.DataFrame()).get("label")
    stats = compute_feature_statistics(train_series, feature_name=feature_name)

    by_dimension = {
        FeatureDiscoveryDimensionKind.INFORMATION_CONTENT: evaluate_information_content(feature_name, stats),
        FeatureDiscoveryDimensionKind.TEMPORAL_STABILITY: evaluate_temporal_stability(feature_name, train_series, stats),
        FeatureDiscoveryDimensionKind.REGIME_STABILITY: evaluate_regime_stability(feature_name, train_series),
        FeatureDiscoveryDimensionKind.DRIFT_BEHAVIOUR: evaluate_drift_behaviour(feature_name, facts.splits),
        FeatureDiscoveryDimensionKind.REDUNDANCY: evaluate_redundancy(feature_name, facts),
        FeatureDiscoveryDimensionKind.COVERAGE: evaluate_coverage(feature_name, train_series, label_series, stats),
        FeatureDiscoveryDimensionKind.AVAILABILITY: evaluate_availability(feature_name, train_series, facts.manifest),
        FeatureDiscoveryDimensionKind.LEAKAGE_SAFETY: evaluate_leakage_safety(feature_name, facts, train_series),
        FeatureDiscoveryDimensionKind.DETERMINISM: evaluate_determinism(feature_name, facts),
        FeatureDiscoveryDimensionKind.REPRODUCIBILITY: evaluate_reproducibility(feature_name, facts),
    }
    dimension_results = tuple(by_dimension[dimension] for dimension in FEATURE_DISCOVERY_DIMENSION_ORDER)
    overall_score = sum(r.score for r in dimension_results) / len(dimension_results)
    return FeatureSignalDiagnostics(feature_name=feature_name, dimension_results=dimension_results, overall_score=overall_score)
