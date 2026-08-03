"""`LabelStability` (Milestone 11, Phase 3, Part C): temporal stability
(rolling balance, rolling entropy, rolling variance, window stability,
expanding stability, availability stability) and regime stability
(comparing a label's own distribution across caller-supplied regime
segments). **Measures only stability -- never predictive power.**
Nothing here computes a correlation, an information coefficient, or any
statistic relating a label to a future outcome; every number is a pure
property of the label's OWN value sequence over time or across regimes.

Regime assignment (bull/bear/sideways/high volatility/low volatility/
macro tightening/macro easing) is always a CALLER-SUPPLIED
`pd.Series` aligned 1:1 with the bundle's own rows -- this module never
computes a regime itself (that would require raw price/macro data this
package deliberately never reads; see `docs/
label_validation_architecture.md`'s dependency-isolation section)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import LabelValidationRequestError
from quant_platform.historical.quality import Severity
from quant_platform.label_validation.distribution import bucket_values
from quant_platform.label_validation.evidence import (
    LabelEvidence,
    LabelValidationDimensionKind,
    make_evidence,
)
from quant_platform.labels.builder import LabelBundle
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = [
    "LABEL_STABILITY_SCHEMA_VERSION",
    "LabelStability",
    "RegimeStabilityResult",
    "TemporalStabilityResult",
    "compute_label_stability",
]

LABEL_STABILITY_SCHEMA_VERSION = 1
_EPSILON = 1e-12


def _encode_buckets(bucketed: pd.Series) -> pd.Series:
    """`pandas.Series.rolling` requires a numeric dtype -- `bucketed` is
    a series of STRING bucket labels (see `distribution.bucket_values`).
    Encodes each distinct label to an integer code (`NaN` stays `NaN`);
    entropy/majority-fraction are invariant to which integer a given
    label happens to receive, so this loses no information the rolling
    computation below needs."""
    codes, _ = pd.factorize(bucketed, use_na_sentinel=True)
    encoded = pd.Series(codes.astype(float), index=bucketed.index)
    return encoded.where(codes != -1)


def _rolling_entropy(bucketed: pd.Series, window_bars: int) -> pd.Series:
    def _entropy(window: np.ndarray) -> float:
        valid = window[~np.isnan(window)]
        if valid.size == 0:
            return np.nan
        _, counts = np.unique(valid, return_counts=True)
        probabilities = counts / valid.size
        return float(-(probabilities * np.log2(probabilities)).sum())

    encoded = _encode_buckets(bucketed)
    return encoded.rolling(window=window_bars, min_periods=max(2, window_bars // 2)).apply(_entropy, raw=True)


def _rolling_majority_fraction(bucketed: pd.Series, window_bars: int) -> pd.Series:
    def _majority(window: np.ndarray) -> float:
        valid = window[~np.isnan(window)]
        if valid.size == 0:
            return np.nan
        _, counts = np.unique(valid, return_counts=True)
        return float(counts.max()) / valid.size

    encoded = _encode_buckets(bucketed)
    return encoded.rolling(window=window_bars, min_periods=max(2, window_bars // 2)).apply(_majority, raw=True)


@dataclass(frozen=True, slots=True)
class TemporalStabilityResult:
    schema_version: int
    label_specification_id: str
    rolling_balance_std: float | None
    """Std of the rolling-window majority-class fraction over time."""
    rolling_entropy_std: float | None
    """Std of the rolling-window entropy over time."""
    rolling_variance_std: float | None
    """Std of the rolling-window std of raw values over time -- "vol of vol"."""
    window_stability_score: float | None
    """`1 - std(per-window means) / (overall std + eps)`, clipped to `[0, 1]` -- higher is more stable."""
    expanding_stability_score: float | None
    """How much the expanding mean has converged by the end of the series -- smaller deviation is more stable."""
    availability_stability: float | None
    """Std of the rolling coverage fraction (non-`NaN` share per window) over time."""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id,
            "rolling_balance_std": self.rolling_balance_std, "rolling_entropy_std": self.rolling_entropy_std,
            "rolling_variance_std": self.rolling_variance_std, "window_stability_score": self.window_stability_score,
            "expanding_stability_score": self.expanding_stability_score, "availability_stability": self.availability_stability,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> TemporalStabilityResult:
        def _optional_float(key: str) -> float | None:
            value = raw.get(key)
            return None if value is None else float(str(value))

        return cls(
            schema_version=int(str(raw["schema_version"])), label_specification_id=str(raw["label_specification_id"]),
            rolling_balance_std=_optional_float("rolling_balance_std"), rolling_entropy_std=_optional_float("rolling_entropy_std"),
            rolling_variance_std=_optional_float("rolling_variance_std"), window_stability_score=_optional_float("window_stability_score"),
            expanding_stability_score=_optional_float("expanding_stability_score"), availability_stability=_optional_float("availability_stability"),
        )


def _compute_temporal_stability(bundle: LabelBundle, *, window_bars: int) -> TemporalStabilityResult:
    spec_id = bundle.specification.label_specification_id
    values = bundle.values
    valid = values.dropna()

    if len(valid) < window_bars:
        return TemporalStabilityResult(
            schema_version=LABEL_STABILITY_SCHEMA_VERSION, label_specification_id=spec_id, rolling_balance_std=None,
            rolling_entropy_std=None, rolling_variance_std=None, window_stability_score=None, expanding_stability_score=None,
            availability_stability=None,
        )

    bucketed = bucket_values(values, label_family=bundle.specification.label_family)
    rolling_entropy = _rolling_entropy(bucketed, window_bars).dropna()
    rolling_balance = _rolling_majority_fraction(bucketed, window_bars).dropna()
    rolling_variance = values.rolling(window=window_bars, min_periods=max(2, window_bars // 2)).std().dropna()
    rolling_coverage = values.notna().rolling(window=window_bars, min_periods=1).mean().dropna()

    overall_std = float(valid.std(ddof=0))
    window_count = max(1, len(valid) // window_bars)
    window_means = [float(w.mean()) for w in np.array_split(valid.to_numpy(), window_count) if len(w) > 0]
    window_stability_score = None
    if len(window_means) >= 2:
        spread = float(np.std(window_means))
        window_stability_score = float(np.clip(1.0 - spread / (overall_std + _EPSILON), 0.0, 1.0))

    expanding_mean = valid.expanding(min_periods=1).mean()
    expanding_stability_score = None
    if len(expanding_mean) >= 2:
        midpoint = len(expanding_mean) // 2
        expanding_stability_score = float(abs(expanding_mean.iloc[-1] - expanding_mean.iloc[midpoint]) / (overall_std + _EPSILON))

    return TemporalStabilityResult(
        schema_version=LABEL_STABILITY_SCHEMA_VERSION, label_specification_id=spec_id,
        rolling_balance_std=(float(rolling_balance.std(ddof=0)) if len(rolling_balance) > 1 else None),
        rolling_entropy_std=(float(rolling_entropy.std(ddof=0)) if len(rolling_entropy) > 1 else None),
        rolling_variance_std=(float(rolling_variance.std(ddof=0)) if len(rolling_variance) > 1 else None),
        window_stability_score=window_stability_score, expanding_stability_score=expanding_stability_score,
        availability_stability=(float(rolling_coverage.std(ddof=0)) if len(rolling_coverage) > 1 else None),
    )


@dataclass(frozen=True, slots=True)
class RegimeStabilityResult:
    schema_version: int
    label_specification_id: str
    per_regime_mean: dict[str, float]
    per_regime_valid_count: dict[str, int]
    cross_regime_mean_spread: float | None
    """`max(per_regime_mean) - min(per_regime_mean)` across regimes with at least one valid observation."""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id,
            "per_regime_mean": self.per_regime_mean, "per_regime_valid_count": self.per_regime_valid_count,
            "cross_regime_mean_spread": self.cross_regime_mean_spread,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RegimeStabilityResult:
        spread_raw = raw.get("cross_regime_mean_spread")
        return cls(
            schema_version=int(str(raw["schema_version"])), label_specification_id=str(raw["label_specification_id"]),
            per_regime_mean={str(k): float(str(v)) for k, v in as_json_dict(raw.get("per_regime_mean") or {}, field_name="per_regime_mean").items()},
            per_regime_valid_count={str(k): int(str(v)) for k, v in as_json_dict(raw.get("per_regime_valid_count") or {}, field_name="per_regime_valid_count").items()},
            cross_regime_mean_spread=(None if spread_raw is None else float(str(spread_raw))),
        )


def _compute_regime_stability(bundle: LabelBundle, regime_assignment: pd.Series) -> RegimeStabilityResult:
    spec_id = bundle.specification.label_specification_id
    if len(regime_assignment) != bundle.row_count:
        raise LabelValidationRequestError(
            f"regime_assignment has {len(regime_assignment)} rows, expected {bundle.row_count} to match the bundle",
            context={"label_specification_id": spec_id},
        )

    frame = pd.DataFrame({"value": bundle.values.to_numpy(), "regime": regime_assignment.to_numpy()})
    per_regime_mean: dict[str, float] = {}
    per_regime_valid_count: dict[str, int] = {}
    for regime, group in frame.groupby("regime"):
        valid = group["value"].dropna()
        per_regime_valid_count[str(regime)] = len(valid)
        if len(valid) > 0:
            per_regime_mean[str(regime)] = float(valid.mean())

    spread = (max(per_regime_mean.values()) - min(per_regime_mean.values())) if len(per_regime_mean) >= 2 else None
    return RegimeStabilityResult(
        schema_version=LABEL_STABILITY_SCHEMA_VERSION, label_specification_id=spec_id, per_regime_mean=per_regime_mean,
        per_regime_valid_count=per_regime_valid_count, cross_regime_mean_spread=spread,
    )


@dataclass(frozen=True, slots=True)
class LabelStability:
    schema_version: int
    label_specification_id: str
    temporal: TemporalStabilityResult
    regime: RegimeStabilityResult | None
    evidence: tuple[LabelEvidence, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id,
            "temporal": self.temporal.to_json_dict(), "regime": (None if self.regime is None else self.regime.to_json_dict()),
            "evidence": [e.to_json_dict() for e in self.evidence],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelStability:
        require_schema_version(raw, supported=LABEL_STABILITY_SCHEMA_VERSION, context="LabelStability")
        regime_raw = raw.get("regime")
        return cls(
            schema_version=LABEL_STABILITY_SCHEMA_VERSION, label_specification_id=str(raw["label_specification_id"]),
            temporal=TemporalStabilityResult.from_json_dict(as_json_dict(raw["temporal"], field_name="temporal")),
            regime=(None if regime_raw is None else RegimeStabilityResult.from_json_dict(as_json_dict(regime_raw, field_name="regime"))),
            evidence=tuple(
                LabelEvidence.from_json_dict(as_json_dict(e, field_name="evidence[]"))
                for e in as_json_list(raw.get("evidence") or [], field_name="evidence")
            ),
        )


def compute_label_stability(bundle: LabelBundle, *, window_bars: int = 20, regime_assignment: pd.Series | None = None) -> LabelStability:
    spec_id = bundle.specification.label_specification_id
    temporal = _compute_temporal_stability(bundle, window_bars=window_bars)
    regime = None if regime_assignment is None else _compute_regime_stability(bundle, regime_assignment)

    evidence: list[LabelEvidence] = []
    if temporal.window_stability_score is not None and temporal.window_stability_score < 0.5:
        evidence.append(make_evidence(
            finding=f"window stability score is low ({temporal.window_stability_score:.4f})",
            evidence=(f"window_bars={window_bars}",), dimension=LabelValidationDimensionKind.TEMPORAL_STABILITY, severity=Severity.WARNING,
            affected_labels=(spec_id,), statistics={"window_stability_score": temporal.window_stability_score},
        ))
    if regime is not None and regime.cross_regime_mean_spread is not None:
        evidence.append(make_evidence(
            finding=f"cross-regime mean spread is {regime.cross_regime_mean_spread:.6f}",
            evidence=(f"per_regime_mean={regime.per_regime_mean}",), dimension=LabelValidationDimensionKind.REGIME_STABILITY,
            severity=Severity.INFO, affected_labels=(spec_id,), statistics={"cross_regime_mean_spread": regime.cross_regime_mean_spread},
        ))

    return LabelStability(
        schema_version=LABEL_STABILITY_SCHEMA_VERSION, label_specification_id=spec_id, temporal=temporal, regime=regime,
        evidence=tuple(evidence),
    )
