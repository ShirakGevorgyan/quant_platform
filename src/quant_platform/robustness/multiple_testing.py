"""Multiple-testing and selection-bias control (Milestone 6, Section 7).

A strategy chosen from many candidates (many parameter settings, many
feature sets, many model families) must never be evaluated as if it were
the ONLY strategy tested -- the more candidates searched, the more likely
ONE of them clears any fixed significance bar by chance alone. This
module provides: (1) a durable `StrategyFamily` record so a selected
candidate always remains traceable to the FULL set it was drawn from,
never silently presented in isolation; (2) three standard p-value
corrections (Bonferroni, Holm, Benjamini-Hochberg); and (3) probabilistic/
deflated Sharpe ratio and minimum-track-record-length estimates, each
with its own explicit, checked assumptions -- computed ONLY when those
assumptions are actually satisfiable from the data on hand, never
fabricated. `MultipleTestingError` is raised (not a silently-omitted
value) when an assumption genuinely cannot be met."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from quant_platform.core.exceptions import MultipleTestingError
from quant_platform.ml.fingerprints import fingerprint_json, is_valid_sha256_hex
from quant_platform.ml.persistence import (
    as_json_list,
    format_utc_timestamp,
    parse_utc_timestamp,
    require_schema_version,
    utc_now,
)
from quant_platform.robustness.models import MultipleTestingCorrectionKind

STRATEGY_FAMILY_SCHEMA_VERSION = 1
MULTIPLE_TESTING_REPORT_SCHEMA_VERSION = 1
_EULER_MASCHERONI = 0.5772156649015329


# --------------------------------------------------------------------------
# StrategyFamily (Section 7)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StrategyFamily:
    """Closure-audit note (performance/resource, Section 13):
    `candidate_backtest_ids` has no upper-bound count check (only non-
    empty + duplicate-free) -- same disclosed operational consideration
    as `BootstrapSpec.repetitions`/`PerturbationSpec.relative_deltas`'s
    own notes; no hard cap invented here."""

    schema_version: int
    family_id: str
    candidate_backtest_ids: tuple[str, ...]
    candidate_experiment_ids: tuple[str, ...]
    candidate_calibration_ids: tuple[str, ...]
    candidate_optimization_identities: tuple[str, ...]
    search_space_identity: str
    selection_metric: str
    eligibility_rules_description: str
    created_at: str

    def __post_init__(self) -> None:
        if not self.candidate_backtest_ids:
            raise MultipleTestingError("StrategyFamily.candidate_backtest_ids must not be empty")
        if len(set(self.candidate_backtest_ids)) != len(self.candidate_backtest_ids):
            raise MultipleTestingError("StrategyFamily.candidate_backtest_ids must not contain duplicates")
        for backtest_id in self.candidate_backtest_ids:
            if not is_valid_sha256_hex(backtest_id):
                raise MultipleTestingError(f"StrategyFamily.candidate_backtest_ids must all be valid sha256 hex ids, got {backtest_id!r}")
        if not self.selection_metric:
            raise MultipleTestingError("StrategyFamily.selection_metric must not be empty")
        parse_utc_timestamp(self.created_at)

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_backtest_ids)

    def to_identity_payload(self) -> dict[str, object]:
        return {
            "candidate_backtest_ids": sorted(self.candidate_backtest_ids), "candidate_experiment_ids": sorted(self.candidate_experiment_ids),
            "candidate_calibration_ids": sorted(self.candidate_calibration_ids), "candidate_optimization_identities": sorted(self.candidate_optimization_identities),
            "search_space_identity": self.search_space_identity, "selection_metric": self.selection_metric,
            "eligibility_rules_description": self.eligibility_rules_description,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "family_id": self.family_id, "candidate_backtest_ids": list(self.candidate_backtest_ids),
            "candidate_experiment_ids": list(self.candidate_experiment_ids), "candidate_calibration_ids": list(self.candidate_calibration_ids),
            "candidate_optimization_identities": list(self.candidate_optimization_identities), "search_space_identity": self.search_space_identity,
            "selection_metric": self.selection_metric, "eligibility_rules_description": self.eligibility_rules_description,
            "candidate_count": self.candidate_count, "created_at": self.created_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> StrategyFamily:
        require_schema_version(raw, supported=STRATEGY_FAMILY_SCHEMA_VERSION, context="StrategyFamily")
        return cls(
            schema_version=STRATEGY_FAMILY_SCHEMA_VERSION, family_id=str(raw["family_id"]),
            candidate_backtest_ids=tuple(str(x) for x in as_json_list(raw["candidate_backtest_ids"], field_name="candidate_backtest_ids")),
            candidate_experiment_ids=tuple(str(x) for x in as_json_list(raw.get("candidate_experiment_ids") or [], field_name="candidate_experiment_ids")),
            candidate_calibration_ids=tuple(str(x) for x in as_json_list(raw.get("candidate_calibration_ids") or [], field_name="candidate_calibration_ids")),
            candidate_optimization_identities=tuple(str(x) for x in as_json_list(raw.get("candidate_optimization_identities") or [], field_name="candidate_optimization_identities")),
            search_space_identity=str(raw["search_space_identity"]), selection_metric=str(raw["selection_metric"]),
            eligibility_rules_description=str(raw.get("eligibility_rules_description", "")), created_at=str(raw["created_at"]),
        )


def build_strategy_family(
    *, candidate_backtest_ids: tuple[str, ...], search_space_identity: str, selection_metric: str,
    candidate_experiment_ids: tuple[str, ...] = (), candidate_calibration_ids: tuple[str, ...] = (),
    candidate_optimization_identities: tuple[str, ...] = (), eligibility_rules_description: str = "",
) -> StrategyFamily:
    payload = {
        "candidate_backtest_ids": sorted(candidate_backtest_ids), "candidate_experiment_ids": sorted(candidate_experiment_ids),
        "candidate_calibration_ids": sorted(candidate_calibration_ids), "candidate_optimization_identities": sorted(candidate_optimization_identities),
        "search_space_identity": search_space_identity, "selection_metric": selection_metric,
        "eligibility_rules_description": eligibility_rules_description, "identity_schema_version": 1,
    }
    family_id = fingerprint_json(payload)
    return StrategyFamily(
        schema_version=STRATEGY_FAMILY_SCHEMA_VERSION, family_id=family_id, candidate_backtest_ids=candidate_backtest_ids,
        candidate_experiment_ids=candidate_experiment_ids, candidate_calibration_ids=candidate_calibration_ids,
        candidate_optimization_identities=candidate_optimization_identities, search_space_identity=search_space_identity,
        selection_metric=selection_metric, eligibility_rules_description=eligibility_rules_description, created_at=format_utc_timestamp(utc_now()),
    )


# --------------------------------------------------------------------------
# P-value corrections (Section 7)
# --------------------------------------------------------------------------
def _validate_p_values(p_values: tuple[float, ...]) -> None:
    if not p_values:
        raise MultipleTestingError("p_values must not be empty")
    for p in p_values:
        if not math.isfinite(p) or not (0.0 <= p <= 1.0):
            raise MultipleTestingError(f"every p-value must be finite and in [0, 1], got {p!r}")


def bonferroni_correction(p_values: tuple[float, ...]) -> tuple[float, ...]:
    """`p_adj_i = min(1, p_i * m)`, order preserved."""
    _validate_p_values(p_values)
    m = len(p_values)
    return tuple(min(1.0, p * m) for p in p_values)


def holm_correction(p_values: tuple[float, ...]) -> tuple[float, ...]:
    """Holm (1979) step-down: sort ascending, adjust `p_(k) * (m-k+1)`
    (1-indexed), then enforce monotonicity by cumulative maximum from the
    smallest p-value upward. Returns adjusted p-values in the ORIGINAL
    input order."""
    _validate_p_values(p_values)
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted_sorted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        raw = p_values[idx] * (m - rank)
        running_max = max(running_max, raw)
        adjusted_sorted[rank] = min(1.0, running_max)
    result = [0.0] * m
    for rank, idx in enumerate(order):
        result[idx] = adjusted_sorted[rank]
    return tuple(result)


def benjamini_hochberg_correction(p_values: tuple[float, ...]) -> tuple[float, ...]:
    """Benjamini-Hochberg (1995) step-up FDR control: sort ascending,
    adjust `p_(k) * m / k` (1-indexed), enforce monotonicity by cumulative
    MINIMUM from the LARGEST p-value downward. Returns adjusted p-values
    in the ORIGINAL input order."""
    _validate_p_values(p_values)
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted_sorted = [0.0] * m
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        raw = p_values[idx] * m / (rank + 1)
        running_min = min(running_min, raw)
        adjusted_sorted[rank] = min(1.0, running_min)
    result = [0.0] * m
    for rank, idx in enumerate(order):
        result[idx] = adjusted_sorted[rank]
    return tuple(result)


_CORRECTION_FUNCTIONS = {
    MultipleTestingCorrectionKind.BONFERRONI: bonferroni_correction,
    MultipleTestingCorrectionKind.HOLM: holm_correction,
    MultipleTestingCorrectionKind.BENJAMINI_HOCHBERG: benjamini_hochberg_correction,
}


def apply_correction(kind: MultipleTestingCorrectionKind, p_values: tuple[float, ...]) -> tuple[float, ...]:
    return _CORRECTION_FUNCTIONS[kind](p_values)


# --------------------------------------------------------------------------
# Probabilistic / deflated Sharpe ratio, minimum track record length
# (Section 7) -- Bailey & Lopez de Prado's formulas, EXACTLY as published.
# --------------------------------------------------------------------------
def _standard_normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _standard_normal_ppf(p: float) -> float:
    """Acklam's rational approximation of the inverse standard normal
    CDF -- accurate to ~1.15e-9, adequate for this module's confidence-
    level lookups (never used where machine-precision inversion matters)."""
    if not (0.0 < p < 1.0):
        raise MultipleTestingError(f"_standard_normal_ppf: p must be in (0, 1), got {p!r}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def _sample_skewness(values: tuple[float, ...]) -> float:
    n = len(values)
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values)
    if stdev == 0.0:
        return 0.0
    return sum(((v - mean) / stdev) ** 3 for v in values) / n


def _sample_kurtosis(values: tuple[float, ...]) -> float:
    """Non-excess (raw) kurtosis -- 3.0 for a normal distribution, matching
    the convention Bailey & Lopez de Prado's PSR/DSR formulas use."""
    n = len(values)
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values)
    if stdev == 0.0:
        return 3.0
    return sum(((v - mean) / stdev) ** 4 for v in values) / n


@dataclass(frozen=True, slots=True)
class ProbabilisticSharpeResult:
    observed_sharpe: float
    benchmark_sharpe: float
    probabilistic_sharpe_ratio: float
    sample_size: int
    skewness: float
    kurtosis: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "observed_sharpe": self.observed_sharpe, "benchmark_sharpe": self.benchmark_sharpe,
            "probabilistic_sharpe_ratio": self.probabilistic_sharpe_ratio, "sample_size": self.sample_size,
            "skewness": self.skewness, "kurtosis": self.kurtosis,
        }


def compute_probabilistic_sharpe_ratio(returns: tuple[float, ...], *, benchmark_sharpe: float = 0.0) -> ProbabilisticSharpeResult:
    """Bailey & Lopez de Prado (2012): `PSR(SR*) = Phi((SR_hat - SR*) *
    sqrt(n-1) / sqrt(1 - skew*SR_hat + (kurt-1)/4*SR_hat^2))`, using the
    NON-annualized, per-period sample Sharpe ratio. Requires `n >= 10`
    (below that, skew/kurtosis estimates are too noisy to trust) --
    raises `MultipleTestingError` otherwise, never a fabricated value."""
    n = len(returns)
    if n < 10:
        raise MultipleTestingError(f"compute_probabilistic_sharpe_ratio: requires at least 10 observations, got {n}")
    stdev = statistics.pstdev(returns)
    if stdev == 0.0:
        raise MultipleTestingError("compute_probabilistic_sharpe_ratio: zero-variance return series -- Sharpe ratio is undefined")
    sr_hat = statistics.fmean(returns) / stdev
    skew = _sample_skewness(returns)
    kurt = _sample_kurtosis(returns)
    denom_sq = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat**2
    if denom_sq <= 0.0:
        raise MultipleTestingError(
            f"compute_probabilistic_sharpe_ratio: the PSR variance term is non-positive ({denom_sq!r}) for this "
            "series' skew/kurtosis/Sharpe combination -- the normal approximation this formula relies on does not "
            "hold here; refusing to report a fabricated value"
        )
    z = (sr_hat - benchmark_sharpe) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    psr = _standard_normal_cdf(z)
    return ProbabilisticSharpeResult(observed_sharpe=sr_hat, benchmark_sharpe=benchmark_sharpe, probabilistic_sharpe_ratio=psr, sample_size=n, skewness=skew, kurtosis=kurt)


@dataclass(frozen=True, slots=True)
class DeflatedSharpeResult:
    observed_sharpe: float
    expected_max_sharpe_under_null: float
    deflated_sharpe_ratio: float
    trials: int
    sharpe_std_across_trials: float
    sample_size: int
    skewness: float
    kurtosis: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "observed_sharpe": self.observed_sharpe, "expected_max_sharpe_under_null": self.expected_max_sharpe_under_null,
            "deflated_sharpe_ratio": self.deflated_sharpe_ratio, "trials": self.trials, "sharpe_std_across_trials": self.sharpe_std_across_trials,
            "sample_size": self.sample_size, "skewness": self.skewness, "kurtosis": self.kurtosis,
        }


def compute_deflated_sharpe_ratio(
    returns: tuple[float, ...], *, sharpe_ratios_across_family: tuple[float, ...],
) -> DeflatedSharpeResult:
    """Bailey & Lopez de Prado (2014): PSR evaluated at the EXPECTED
    MAXIMUM Sharpe ratio achievable by chance across `N` independent
    trials, `SR*_max ~= sigma_SR * [(1-gamma)*Phi^-1(1-1/N) +
    gamma*Phi^-1(1-1/(N*e))]`, where `sigma_SR` is the standard deviation
    of Sharpe ratios OBSERVED ACROSS THE ACTUAL FAMILY of candidates
    (never assumed/fabricated -- the caller must supply the family's own
    real Sharpe ratios). Requires at least 2 candidates in the family (to
    estimate `sigma_SR` at all) and `N >= 2`; raises `MultipleTestingError`
    otherwise -- this is the explicit "fail closed rather than fabricate"
    path Section 7 requires when assumptions are not available."""
    n_trials = len(sharpe_ratios_across_family)
    if n_trials < 2:
        raise MultipleTestingError(
            f"compute_deflated_sharpe_ratio: requires >= 2 candidate Sharpe ratios to estimate the null "
            f"distribution's standard deviation, got {n_trials} -- refusing to fabricate an assumed sigma_SR"
        )
    sigma_sr = statistics.pstdev(sharpe_ratios_across_family)
    if sigma_sr == 0.0:
        raise MultipleTestingError("compute_deflated_sharpe_ratio: zero standard deviation of Sharpe ratios across the family -- expected-max-Sharpe is undefined")
    expected_max_sr = sigma_sr * ((1.0 - _EULER_MASCHERONI) * _standard_normal_ppf(1.0 - 1.0 / n_trials) + _EULER_MASCHERONI * _standard_normal_ppf(1.0 - 1.0 / (n_trials * math.e)))
    psr = compute_probabilistic_sharpe_ratio(returns, benchmark_sharpe=expected_max_sr)
    return DeflatedSharpeResult(
        observed_sharpe=psr.observed_sharpe, expected_max_sharpe_under_null=expected_max_sr, deflated_sharpe_ratio=psr.probabilistic_sharpe_ratio,
        trials=n_trials, sharpe_std_across_trials=sigma_sr, sample_size=psr.sample_size, skewness=psr.skewness, kurtosis=psr.kurtosis,
    )


def compute_minimum_track_record_length(returns: tuple[float, ...], *, benchmark_sharpe: float = 0.0, confidence_level: float = 0.95) -> float:
    """Bailey & Lopez de Prado (2012): `MinTRL = 1 + (1 - skew*SR +
    (kurt-1)/4*SR^2) * (Z_alpha / (SR - SR*))^2`. Requires `SR > SR*`
    strictly (otherwise the required track record length is infinite/
    undefined) -- raises `MultipleTestingError` rather than returning
    `inf`."""
    n = len(returns)
    if n < 10:
        raise MultipleTestingError(f"compute_minimum_track_record_length: requires at least 10 observations, got {n}")
    stdev = statistics.pstdev(returns)
    if stdev == 0.0:
        raise MultipleTestingError("compute_minimum_track_record_length: zero-variance return series -- Sharpe ratio is undefined")
    sr_hat = statistics.fmean(returns) / stdev
    if sr_hat <= benchmark_sharpe:
        raise MultipleTestingError(
            f"compute_minimum_track_record_length: observed Sharpe ({sr_hat!r}) does not exceed benchmark_sharpe "
            f"({benchmark_sharpe!r}) -- the required track record length is undefined (would be infinite)"
        )
    skew = _sample_skewness(returns)
    kurt = _sample_kurtosis(returns)
    z_alpha = _standard_normal_ppf(confidence_level)
    variance_term = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat**2
    if variance_term <= 0.0:
        raise MultipleTestingError(f"compute_minimum_track_record_length: the variance term is non-positive ({variance_term!r}) for this series -- refusing to fabricate a value")
    return 1.0 + variance_term * (z_alpha / (sr_hat - benchmark_sharpe)) ** 2


__all__ = [
    "MULTIPLE_TESTING_REPORT_SCHEMA_VERSION",
    "STRATEGY_FAMILY_SCHEMA_VERSION",
    "DeflatedSharpeResult",
    "ProbabilisticSharpeResult",
    "StrategyFamily",
    "apply_correction",
    "benjamini_hochberg_correction",
    "bonferroni_correction",
    "build_strategy_family",
    "compute_deflated_sharpe_ratio",
    "compute_minimum_track_record_length",
    "compute_probabilistic_sharpe_ratio",
    "holm_correction",
]
