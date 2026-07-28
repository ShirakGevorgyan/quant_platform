"""Reliability diagnostics (Milestone 4E, Section 11) -- structured,
durable bin-level calibration diagnostics, independent of any single
scalar metric.

BIN-ASSIGNMENT SEMANTICS (EXPLICIT, PER SECTION 10/11)
--------------------------------------------------------------------------
Equal-width bins: `n_bins` equal-width intervals over `[0, 1]`. Interval
boundaries are LEFT-closed/right-open (`[k/n, (k+1)/n)`) for every bin
except the last, which is closed on both ends (`[(n-1)/n, 1.0]`) so a
probability of exactly `1.0` always lands in a real bin rather than
overflowing past it -- the same convention `sklearn.calibration.
calibration_curve` uses.

Equal-frequency bins: `n_bins` quantile-edge intervals computed from the
ACTUAL predicted-probability distribution being diagnosed (never a fixed
a-priori grid). When many predicted probabilities repeat (a common
occurrence for a coarse or under-trained model), adjacent quantile edges
can collapse to the same value -- attempting to build a zero-width bin
from them would either divide by zero or silently manufacture a nominal
bin with no distinguishable interval. Collapsed edges are deduplicated
before bin construction, which can reduce the ACTUAL bin count below the
REQUESTED `n_bins`; `ReliabilityReport.actual_n_bins` records the true
count and `EqualFrequencyCollapseNote` documents why, rather than the
caller silently receiving fewer bins than requested with no explanation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quant_platform.calibration.models import BinningStrategy
from quant_platform.calibration.specs import ReliabilityBinningSpec
from quant_platform.core.exceptions import CalibrationDataError
from quant_platform.ml.persistence import as_json_list, require_schema_version

RELIABILITY_REPORT_SCHEMA_VERSION = 1
_Z_95 = 1.959963984540054
"""Standard normal critical value for a 95% Wilson score confidence
interval -- a fixed, documented constant, never re-derived per call."""


def _wilson_interval(successes: int, n: int, *, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion -- well-behaved
    (unlike the naive normal approximation) at proportions near 0 or 1,
    which reliability bins routinely produce."""
    if n == 0:
        raise CalibrationDataError("_wilson_interval requires n >= 1")
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half_width = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    bin_index: int
    lower_bound: float
    upper_bound: float
    sample_count: int
    mean_predicted_probability: float | None
    empirical_positive_rate: float | None
    calibration_gap: float | None
    confidence_interval_low: float | None
    confidence_interval_high: float | None
    is_empty: bool

    def __post_init__(self) -> None:
        if self.bin_index < 0:
            raise CalibrationDataError(f"ReliabilityBin.bin_index must be >= 0, got {self.bin_index}")
        if not (0.0 <= self.lower_bound <= self.upper_bound <= 1.0):
            raise CalibrationDataError(
                f"ReliabilityBin bounds must satisfy 0 <= lower_bound <= upper_bound <= 1, "
                f"got [{self.lower_bound}, {self.upper_bound}]"
            )
        if self.sample_count < 0:
            raise CalibrationDataError(f"ReliabilityBin.sample_count must be >= 0, got {self.sample_count}")
        if self.is_empty != (self.sample_count == 0):
            raise CalibrationDataError("ReliabilityBin.is_empty must be exactly (sample_count == 0)")
        if self.is_empty:
            for name, value in (
                ("mean_predicted_probability", self.mean_predicted_probability),
                ("empirical_positive_rate", self.empirical_positive_rate), ("calibration_gap", self.calibration_gap),
            ):
                if value is not None:
                    raise CalibrationDataError(f"ReliabilityBin.{name} must be None for an empty bin")
        for name, value in (
            ("mean_predicted_probability", self.mean_predicted_probability), ("empirical_positive_rate", self.empirical_positive_rate),
            ("calibration_gap", self.calibration_gap), ("confidence_interval_low", self.confidence_interval_low),
            ("confidence_interval_high", self.confidence_interval_high),
        ):
            if value is not None and not math.isfinite(value):
                raise CalibrationDataError(f"ReliabilityBin.{name} must be finite if set, got {value!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "bin_index": self.bin_index, "lower_bound": self.lower_bound, "upper_bound": self.upper_bound,
            "sample_count": self.sample_count, "mean_predicted_probability": self.mean_predicted_probability,
            "empirical_positive_rate": self.empirical_positive_rate, "calibration_gap": self.calibration_gap,
            "confidence_interval_low": self.confidence_interval_low, "confidence_interval_high": self.confidence_interval_high,
            "is_empty": self.is_empty,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ReliabilityBin:
        def _opt(key: str) -> float | None:
            return None if raw.get(key) is None else float(str(raw[key]))

        return cls(
            bin_index=int(str(raw["bin_index"])), lower_bound=float(str(raw["lower_bound"])), upper_bound=float(str(raw["upper_bound"])),
            sample_count=int(str(raw["sample_count"])), mean_predicted_probability=_opt("mean_predicted_probability"),
            empirical_positive_rate=_opt("empirical_positive_rate"), calibration_gap=_opt("calibration_gap"),
            confidence_interval_low=_opt("confidence_interval_low"), confidence_interval_high=_opt("confidence_interval_high"),
            is_empty=bool(raw["is_empty"]),
        )


@dataclass(frozen=True, slots=True)
class ReliabilityReport:
    schema_version: int
    binning_spec: ReliabilityBinningSpec
    bins: tuple[ReliabilityBin, ...]
    requested_n_bins: int
    actual_n_bins: int
    collapsed_edges_note: str | None
    overall_positive_rate: float
    n_samples: int

    def __post_init__(self) -> None:
        if self.actual_n_bins != len(self.bins):
            raise CalibrationDataError("ReliabilityReport.actual_n_bins must equal len(bins)")
        if self.actual_n_bins > self.requested_n_bins:
            raise CalibrationDataError("ReliabilityReport.actual_n_bins must never exceed requested_n_bins")
        if self.actual_n_bins < self.requested_n_bins and self.collapsed_edges_note is None:
            raise CalibrationDataError(
                "ReliabilityReport: actual_n_bins < requested_n_bins requires a non-None collapsed_edges_note"
            )
        if not (0.0 <= self.overall_positive_rate <= 1.0):
            raise CalibrationDataError(f"ReliabilityReport.overall_positive_rate must be in [0, 1], got {self.overall_positive_rate}")
        if self.n_samples < 1:
            raise CalibrationDataError(f"ReliabilityReport.n_samples must be >= 1, got {self.n_samples}")
        total_binned = sum(b.sample_count for b in self.bins)
        if total_binned != self.n_samples:
            raise CalibrationDataError(
                f"ReliabilityReport: sum of bin sample_count ({total_binned}) does not equal n_samples ({self.n_samples})"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "binning_spec": self.binning_spec.to_json_dict(),
            "bins": [b.to_json_dict() for b in self.bins], "requested_n_bins": self.requested_n_bins,
            "actual_n_bins": self.actual_n_bins, "collapsed_edges_note": self.collapsed_edges_note,
            "overall_positive_rate": self.overall_positive_rate, "n_samples": self.n_samples,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ReliabilityReport:
        require_schema_version(raw, supported=RELIABILITY_REPORT_SCHEMA_VERSION, context="ReliabilityReport")
        binning_spec_raw = raw["binning_spec"]
        if not isinstance(binning_spec_raw, dict):
            raise CalibrationDataError("ReliabilityReport.binning_spec must be a JSON object")
        return cls(
            schema_version=RELIABILITY_REPORT_SCHEMA_VERSION, binning_spec=ReliabilityBinningSpec.from_json_dict(binning_spec_raw),
            bins=tuple(ReliabilityBin.from_json_dict(b) for b in as_json_list(raw["bins"], field_name="bins")),
            requested_n_bins=int(str(raw["requested_n_bins"])), actual_n_bins=int(str(raw["actual_n_bins"])),
            collapsed_edges_note=(None if raw.get("collapsed_edges_note") is None else str(raw["collapsed_edges_note"])),
            overall_positive_rate=float(str(raw["overall_positive_rate"])), n_samples=int(str(raw["n_samples"])),
        )


def _equal_width_edges(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def _equal_frequency_edges(probabilities: np.ndarray, n_bins: int) -> tuple[np.ndarray, str | None]:
    raw_edges = np.quantile(probabilities, np.linspace(0.0, 1.0, n_bins + 1))
    raw_edges[0], raw_edges[-1] = 0.0, 1.0
    deduped = [raw_edges[0]]
    for edge in raw_edges[1:]:
        if edge > deduped[-1]:
            deduped.append(edge)
    edges = np.asarray(deduped, dtype="float64")
    actual_bins = len(edges) - 1
    note = None
    if actual_bins < n_bins:
        note = (
            f"Requested {n_bins} equal-frequency bins, but {n_bins - actual_bins} adjacent quantile edge(s) "
            f"collapsed (identical predicted probabilities span multiple quantile boundaries) -- {actual_bins} "
            "distinguishable bin(s) were used instead of manufacturing a zero-width bin"
        )
    return edges, note


def compute_reliability_bins(
    probabilities: np.ndarray, labels: np.ndarray, *, spec: ReliabilityBinningSpec,
) -> ReliabilityReport:
    """Builds one `ReliabilityReport` from parallel `probabilities`/
    `labels` arrays (both already validated finite/in-range by the
    caller's `RawPredictionSet`/calibrated-output contract -- this
    function re-validates defensively rather than trusting that)."""
    probs = np.asarray(probabilities, dtype="float64")
    labs = np.asarray(labels, dtype="float64")
    if probs.shape != labs.shape:
        raise CalibrationDataError(f"probabilities shape {probs.shape} does not match labels shape {labs.shape}")
    if probs.ndim != 1 or len(probs) == 0:
        raise CalibrationDataError("probabilities/labels must be non-empty 1-dimensional arrays")
    if not np.all(np.isfinite(probs)):
        raise CalibrationDataError("probabilities contains non-finite value(s)")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise CalibrationDataError("probabilities contains value(s) outside [0, 1]")
    if not np.all(np.isin(labs, (0.0, 1.0))):
        raise CalibrationDataError("labels must be binary (0.0/1.0) valued")

    if spec.strategy is BinningStrategy.EQUAL_WIDTH:
        edges = _equal_width_edges(spec.n_bins)
        collapse_note = None
    else:
        edges, collapse_note = _equal_frequency_edges(probs, spec.n_bins)

    n_actual_bins = len(edges) - 1
    bin_index_per_sample = np.clip(np.searchsorted(edges, probs, side="right") - 1, 0, n_actual_bins - 1)

    bins: list[ReliabilityBin] = []
    for i in range(n_actual_bins):
        mask = bin_index_per_sample == i
        count = int(mask.sum())
        if count == 0:
            bins.append(ReliabilityBin(
                bin_index=i, lower_bound=float(edges[i]), upper_bound=float(edges[i + 1]), sample_count=0,
                mean_predicted_probability=None, empirical_positive_rate=None, calibration_gap=None,
                confidence_interval_low=None, confidence_interval_high=None, is_empty=True,
            ))
            continue
        bin_probs = probs[mask]
        bin_labels = labs[mask]
        mean_pred = float(bin_probs.mean())
        positive_rate = float(bin_labels.mean())
        ci_low, ci_high = _wilson_interval(int(bin_labels.sum()), count)
        bins.append(ReliabilityBin(
            bin_index=i, lower_bound=float(edges[i]), upper_bound=float(edges[i + 1]), sample_count=count,
            mean_predicted_probability=mean_pred, empirical_positive_rate=positive_rate,
            calibration_gap=abs(mean_pred - positive_rate), confidence_interval_low=ci_low, confidence_interval_high=ci_high,
            is_empty=False,
        ))

    return ReliabilityReport(
        schema_version=RELIABILITY_REPORT_SCHEMA_VERSION, binning_spec=spec, bins=tuple(bins),
        requested_n_bins=spec.n_bins, actual_n_bins=n_actual_bins, collapsed_edges_note=collapse_note,
        overall_positive_rate=float(labs.mean()), n_samples=len(probs),
    )


__all__ = [
    "RELIABILITY_REPORT_SCHEMA_VERSION",
    "ReliabilityBin",
    "ReliabilityReport",
    "compute_reliability_bins",
]
