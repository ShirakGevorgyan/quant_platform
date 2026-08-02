"""`FeatureStatistics` (Milestone 11, Phase 2, Part 1): deterministic,
descriptive per-feature statistics only -- variance, entropy, missing/
zero/constant/near-constant ratios, cardinality, unique ratio, and
rolling variants of variance/entropy/missingness. Every number here is
a closed-form, reproducible computation over already-materialized
values; nothing here fits, trains, or estimates a model of any kind.

Reused, never reimplemented, by `diagnostics.py`'s Information Content
dimension (the direct consumer) and referenced by several other
dimensions (Coverage's missing-ratio, Redundancy's cardinality checks)
so the same numbers are never computed twice for one feature."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.ml.persistence import require_schema_version

__all__ = ["FEATURE_STATISTICS_SCHEMA_VERSION", "FeatureStatistics", "compute_feature_statistics", "rolling_window_size", "shannon_entropy"]

FEATURE_STATISTICS_SCHEMA_VERSION = 2
_EXACT_ENTROPY_MAX_CATEGORIES = 50
_ENTROPY_BINS = 20
_ROLLING_WINDOW_MIN = 10


def rolling_window_size(n: int) -> int:
    return max(_ROLLING_WINDOW_MIN, n // 20)


def shannon_entropy(values: np.ndarray, *, bins: int = _ENTROPY_BINS) -> float:
    """Shannon entropy in NATS. Distributions with at most
    `_EXACT_ENTROPY_MAX_CATEGORIES` distinct values are treated as
    categorical (entropy over exact value counts); larger cardinalities
    are binned into `bins` quantile-edge bins first (the same binning
    strategy `features.drift.population_stability_index` uses, for a
    consistent notion of "bin" across this package). Returns `0.0` for
    fewer than 2 values (a single value, or none, carries zero entropy
    by definition -- not `NaN`, so downstream arithmetic never has to
    special-case an empty feature)."""
    clean = values[~np.isnan(values)] if np.issubdtype(values.dtype, np.floating) else values
    if len(clean) < 2:
        return 0.0

    distinct = np.unique(clean)
    if len(distinct) <= 1:
        return 0.0
    if len(distinct) <= _EXACT_ENTROPY_MAX_CATEGORIES:
        _, counts = np.unique(clean, return_counts=True)
    else:
        edges = np.unique(np.quantile(clean, np.linspace(0.0, 1.0, bins + 1)))
        if len(edges) < 2:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(clean, bins=edges)
        counts = counts[counts > 0]

    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


@dataclass(frozen=True, slots=True)
class FeatureStatistics:
    feature_name: str
    total_rows: int
    count: int
    """Non-null observation count."""
    mean: float
    std: float
    variance: float
    minimum: float
    maximum: float
    missing_ratio: float
    infinite_ratio: float
    """Fraction of non-null values that are `+inf`/`-inf`. All the other statistics below (mean, std,
    variance, cardinality, entropy, rolling variants, ...) are computed over the FINITE-only subset --
    an infinite value would otherwise poison `mean`/`std`/rolling-window arithmetic with `inf - inf =
    NaN` (a real defect found and fixed during this package's own adversarial testing)."""
    zero_ratio: float
    constant_ratio: float
    """`1.0` if every non-null value is identical (std == 0 or count <= 1), else `0.0`."""
    near_constant_ratio: float
    """Fraction of non-null values equal to the single most common value (the mode) -- a continuous
    "how close to constant" measure, distinct from `constant_ratio`'s strict boolean."""
    cardinality: int
    """Distinct non-null value count."""
    effective_cardinality: float
    """`exp(entropy)` -- the "perplexity": how many EQUALLY-LIKELY categories would produce this same
    entropy. Always `<= cardinality`; equals `cardinality` only for a perfectly uniform distribution."""
    unique_ratio: float
    entropy: float
    rolling_variance_mean: float
    rolling_variance_cv: float
    """Coefficient of variation OF the rolling-variance series itself -- how much the feature's own
    local variance fluctuates over time, not the variance of the raw values."""
    rolling_entropy_mean: float
    rolling_entropy_cv: float
    rolling_missingness_mean: float
    rolling_missingness_max_swing: float
    """`max - min` of the rolling null-fraction series."""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": FEATURE_STATISTICS_SCHEMA_VERSION, "feature_name": self.feature_name, "total_rows": self.total_rows,
            "count": self.count, "mean": self.mean, "std": self.std, "variance": self.variance, "minimum": self.minimum,
            "maximum": self.maximum, "missing_ratio": self.missing_ratio, "infinite_ratio": self.infinite_ratio, "zero_ratio": self.zero_ratio,
            "constant_ratio": self.constant_ratio, "near_constant_ratio": self.near_constant_ratio, "cardinality": self.cardinality,
            "effective_cardinality": self.effective_cardinality, "unique_ratio": self.unique_ratio, "entropy": self.entropy,
            "rolling_variance_mean": self.rolling_variance_mean, "rolling_variance_cv": self.rolling_variance_cv,
            "rolling_entropy_mean": self.rolling_entropy_mean, "rolling_entropy_cv": self.rolling_entropy_cv,
            "rolling_missingness_mean": self.rolling_missingness_mean, "rolling_missingness_max_swing": self.rolling_missingness_max_swing,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureStatistics:
        require_schema_version(raw, supported=FEATURE_STATISTICS_SCHEMA_VERSION, context="FeatureStatistics")
        return cls(
            feature_name=str(raw["feature_name"]), total_rows=int(str(raw["total_rows"])), count=int(str(raw["count"])),
            mean=float(str(raw["mean"])), std=float(str(raw["std"])), variance=float(str(raw["variance"])),
            minimum=float(str(raw["minimum"])), maximum=float(str(raw["maximum"])), missing_ratio=float(str(raw["missing_ratio"])),
            infinite_ratio=float(str(raw["infinite_ratio"])), zero_ratio=float(str(raw["zero_ratio"])), constant_ratio=float(str(raw["constant_ratio"])),
            near_constant_ratio=float(str(raw["near_constant_ratio"])), cardinality=int(str(raw["cardinality"])),
            effective_cardinality=float(str(raw["effective_cardinality"])), unique_ratio=float(str(raw["unique_ratio"])),
            entropy=float(str(raw["entropy"])), rolling_variance_mean=float(str(raw["rolling_variance_mean"])),
            rolling_variance_cv=float(str(raw["rolling_variance_cv"])), rolling_entropy_mean=float(str(raw["rolling_entropy_mean"])),
            rolling_entropy_cv=float(str(raw["rolling_entropy_cv"])), rolling_missingness_mean=float(str(raw["rolling_missingness_mean"])),
            rolling_missingness_max_swing=float(str(raw["rolling_missingness_max_swing"])),
        )


def _coefficient_of_variation(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) < 2:
        return 0.0
    mean = float(clean.mean())
    if mean == 0:
        return 0.0
    return float(clean.std() / mean)


def compute_feature_statistics(series: pd.Series, *, feature_name: str) -> FeatureStatistics:
    """`series` must already be in its dataset's own chronological row
    order (every split DataFrame `ResearchDatasetBuilder` produces
    already is) -- rolling statistics are order-sensitive by
    definition."""
    total_rows = len(series)
    clean = series.dropna()
    count = len(clean)
    values = clean.to_numpy(dtype="float64") if count else np.array([], dtype="float64")

    missing_ratio = 1.0 - (count / total_rows) if total_rows else 0.0
    if count == 0:
        return FeatureStatistics(
            feature_name=feature_name, total_rows=total_rows, count=0, mean=0.0, std=0.0, variance=0.0, minimum=0.0, maximum=0.0,
            missing_ratio=missing_ratio, infinite_ratio=0.0, zero_ratio=0.0, constant_ratio=1.0, near_constant_ratio=1.0, cardinality=0,
            effective_cardinality=0.0, unique_ratio=0.0, entropy=0.0, rolling_variance_mean=0.0, rolling_variance_cv=0.0,
            rolling_entropy_mean=0.0, rolling_entropy_cv=0.0, rolling_missingness_mean=missing_ratio, rolling_missingness_max_swing=0.0,
        )

    finite_mask = np.isfinite(values)
    infinite_ratio = float((~finite_mask).sum() / count)
    finite_values = values[finite_mask]
    finite_count = len(finite_values)
    # `+/-inf` poisons mean/std/rolling-window arithmetic with `inf - inf = NaN` -- every statistic
    # below (except missing_ratio/infinite_ratio themselves, which describe the RAW series) is
    # computed over the finite-only subset, never the raw non-null values.
    finite_series = series.mask(np.isinf(series), other=np.nan)

    if finite_count == 0:
        return FeatureStatistics(
            feature_name=feature_name, total_rows=total_rows, count=count, mean=0.0, std=0.0, variance=0.0, minimum=0.0, maximum=0.0,
            missing_ratio=missing_ratio, infinite_ratio=infinite_ratio, zero_ratio=0.0, constant_ratio=0.0, near_constant_ratio=1.0,
            cardinality=0, effective_cardinality=0.0, unique_ratio=0.0, entropy=0.0, rolling_variance_mean=0.0, rolling_variance_cv=0.0,
            rolling_entropy_mean=0.0, rolling_entropy_cv=0.0, rolling_missingness_mean=missing_ratio, rolling_missingness_max_swing=0.0,
        )

    mean, std = float(finite_values.mean()), float(finite_values.std())
    variance = std**2
    zero_ratio = float((finite_values == 0).sum() / finite_count)
    distinct_values, distinct_counts = np.unique(finite_values, return_counts=True)
    cardinality = len(distinct_values)
    mode_count = int(distinct_counts.max())
    constant_ratio = 1.0 if cardinality <= 1 else 0.0
    near_constant_ratio = mode_count / finite_count
    unique_ratio = cardinality / finite_count
    entropy = shannon_entropy(finite_values)
    effective_cardinality = float(np.exp(entropy))

    window = rolling_window_size(total_rows)
    if window < total_rows:
        rolling_std = finite_series.rolling(window=window, min_periods=window).std()
        rolling_variance_series = (rolling_std**2).dropna()
        rolling_variance_mean = float(rolling_variance_series.mean()) if len(rolling_variance_series) else 0.0
        rolling_variance_cv = _coefficient_of_variation(rolling_variance_series)

        rolling_entropy_series = finite_series.rolling(window=window, min_periods=window).apply(
            lambda w: shannon_entropy(w[~np.isnan(w)]), raw=True,
        ).dropna()
        rolling_entropy_mean = float(rolling_entropy_series.mean()) if len(rolling_entropy_series) else 0.0
        rolling_entropy_cv = _coefficient_of_variation(rolling_entropy_series)

        rolling_missingness_series = series.isna().rolling(window=window, min_periods=window).mean().dropna()
        rolling_missingness_mean = float(rolling_missingness_series.mean()) if len(rolling_missingness_series) else missing_ratio
        rolling_missingness_max_swing = (
            float(rolling_missingness_series.max() - rolling_missingness_series.min()) if len(rolling_missingness_series) else 0.0
        )
    else:
        rolling_variance_mean, rolling_variance_cv = variance, 0.0
        rolling_entropy_mean, rolling_entropy_cv = entropy, 0.0
        rolling_missingness_mean, rolling_missingness_max_swing = missing_ratio, 0.0

    return FeatureStatistics(
        feature_name=feature_name, total_rows=total_rows, count=count, mean=mean, std=std, variance=variance,
        minimum=float(finite_values.min()), maximum=float(finite_values.max()), missing_ratio=missing_ratio, infinite_ratio=infinite_ratio,
        zero_ratio=zero_ratio, constant_ratio=constant_ratio, near_constant_ratio=near_constant_ratio, cardinality=cardinality,
        effective_cardinality=effective_cardinality, unique_ratio=unique_ratio, entropy=entropy,
        rolling_variance_mean=rolling_variance_mean, rolling_variance_cv=rolling_variance_cv,
        rolling_entropy_mean=rolling_entropy_mean, rolling_entropy_cv=rolling_entropy_cv,
        rolling_missingness_mean=rolling_missingness_mean, rolling_missingness_max_swing=rolling_missingness_max_swing,
    )
