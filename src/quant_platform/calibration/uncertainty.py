"""Uncertainty framework (Milestone 4E, Section 16) -- transparent,
explicitly-scoped proxies, never a claim of exact Bayesian uncertainty.

FOUR CATEGORIES, FIVE PROXIES
--------------------------------------------------------------------------
Section 16 separates aleatoric/observational, epistemic/model,
calibration, and decision uncertainty conceptually; this module
implements the five CONCRETE, transparent proxies Section 16 A-E asks
for and lets the caller (`calibration.fitting`) decide which proxy
serves which conceptual category for its own report:

  A. `entropy_component`             -- aleatoric-flavored (spread of the
                                         predicted distribution itself)
  B. `margin_component`               -- decision uncertainty (nearness
                                         to the decision threshold)
  C. `model_disagreement_component`   -- epistemic (inner-fold model
                                         ensemble disagreement)
  D. `calibrator_disagreement_component` -- calibration uncertainty
                                         (candidate-method disagreement)
  E. `bin_support_uncertainty_component` -- calibration uncertainty
                                         (how much reliability evidence
                                         backs this probability region)

Every proxy is a plain, pure, independently testable function of
already-computed numbers -- none of them reach into a model, a
`RawPredictionSet`, or a reliability report themselves."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from quant_platform.calibration.models import ScoreProvenance
from quant_platform.calibration.specs import UncertaintySpec
from quant_platform.core.exceptions import UncertaintyPolicyError
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

UNCERTAINTY_RESULT_SCHEMA_VERSION = 1
_MAX_BINARY_STD = 0.5
"""The maximum possible standard deviation of a set of values confined to
`{0.0, 1.0}` (half at each extreme) -- the normalizing denominator for
`model_disagreement_component`/`calibrator_disagreement_component`, which
in practice see continuous probabilities in `[0, 1]`, a strictly narrower
range, so this is a conservative (never-exceeded) upper bound."""


def entropy_component(probability: float) -> float:
    """Binary Shannon entropy, normalized by its own maximum (1 bit, at
    `probability == 0.5`) to `[0, 1]`. `0.0` at `probability in {0, 1}`
    (a fully determined outcome), `1.0` at `probability == 0.5`."""
    if not (0.0 <= probability <= 1.0):
        raise UncertaintyPolicyError(f"entropy_component: probability must be in [0, 1], got {probability!r}")
    if probability in (0.0, 1.0):
        return 0.0
    h = -(probability * math.log2(probability) + (1.0 - probability) * math.log2(1.0 - probability))
    return max(0.0, min(1.0, h))


def margin_component(probability: float, threshold: float) -> float:
    """`1.0` exactly at the decision threshold (maximally uncertain
    decision), `0.0` at the probability furthest from it -- the
    complement of `calibration.confidence.distance_from_threshold_component`."""
    from quant_platform.calibration.confidence import distance_from_threshold_component

    return 1.0 - distance_from_threshold_component(probability, threshold)


def _disagreement(values: Sequence[float], *, field_name: str) -> float:
    if len(values) < 2:
        raise UncertaintyPolicyError(f"{field_name} requires at least 2 values to measure disagreement, got {len(values)}")
    for v in values:
        if not (0.0 <= v <= 1.0):
            raise UncertaintyPolicyError(f"{field_name}: every value must be in [0, 1], got {v!r}")
    std = statistics.pstdev(values)
    return max(0.0, min(1.0, std / _MAX_BINARY_STD))


def model_disagreement_component(predictions: Sequence[float]) -> float:
    """Section 16: "Where multiple inner models are used for uncertainty
    ... prediction order must be deterministic; model identities must be
    persisted; aggregation must validate identical class ordering." This
    function only computes the numeric disagreement; the caller
    (`calibration.fitting`) is responsible for the ordering/identity/
    class-ordering invariants before calling it -- see that module."""
    return _disagreement(predictions, field_name="model_disagreement_component")


def calibrator_disagreement_component(calibrated_probabilities: Sequence[float]) -> float:
    return _disagreement(calibrated_probabilities, field_name="calibrator_disagreement_component")


def bin_support_uncertainty_component(bin_sample_count: int, *, minimum_support: int) -> float:
    """`0.0` once `bin_sample_count >= minimum_support`; scales linearly
    up to `1.0` at `bin_sample_count == 0`."""
    if bin_sample_count < 0:
        raise UncertaintyPolicyError(f"bin_support_uncertainty_component: bin_sample_count must be >= 0, got {bin_sample_count}")
    if minimum_support < 1:
        raise UncertaintyPolicyError(f"bin_support_uncertainty_component: minimum_support must be >= 1, got {minimum_support}")
    return max(0.0, 1.0 - (bin_sample_count / minimum_support))


@dataclass(frozen=True, slots=True)
class UncertaintyResult:
    total_uncertainty: float
    components: Mapping[str, float]
    component_availability: Mapping[str, bool]
    reason_codes: tuple[str, ...]
    provenance: ScoreProvenance
    interpretation: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.total_uncertainty) or not (0.0 <= self.total_uncertainty <= 1.0):
            raise UncertaintyPolicyError(f"UncertaintyResult.total_uncertainty must be a finite value in [0, 1], got {self.total_uncertainty!r}")
        for name, value in self.components.items():
            if not math.isfinite(value):
                raise UncertaintyPolicyError(f"UncertaintyResult.components[{name!r}] must be finite, got {value!r}")
        if not self.interpretation:
            raise UncertaintyPolicyError("UncertaintyResult.interpretation must not be empty")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": UNCERTAINTY_RESULT_SCHEMA_VERSION, "total_uncertainty": self.total_uncertainty,
            "components": dict(sorted(self.components.items())), "component_availability": dict(sorted(self.component_availability.items())),
            "reason_codes": list(self.reason_codes), "provenance": self.provenance.value, "interpretation": self.interpretation,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> UncertaintyResult:
        require_schema_version(raw, supported=UNCERTAINTY_RESULT_SCHEMA_VERSION, context="UncertaintyResult")
        return cls(
            total_uncertainty=float(str(raw["total_uncertainty"])),
            components={str(k): float(v) for k, v in as_json_dict(raw["components"], field_name="components").items()},
            component_availability={str(k): bool(v) for k, v in as_json_dict(raw["component_availability"], field_name="component_availability").items()},
            reason_codes=tuple(str(c) for c in as_json_list(raw["reason_codes"], field_name="reason_codes")),
            provenance=ScoreProvenance(raw["provenance"]), interpretation=str(raw["interpretation"]),
        )


_INTERPRETATIONS: dict[str, str] = {
    "entropy": "spread of the predicted probability distribution itself (aleatoric-flavored)",
    "margin": "nearness of the calibrated probability to the decision threshold (decision uncertainty)",
    "model_disagreement": "disagreement across inner-fold models trained on different training-side data (epistemic proxy)",
    "calibrator_disagreement": "disagreement across candidate calibration methods (calibration uncertainty)",
    "bin_support": "how little reliability-diagnostic evidence backs this probability region (calibration uncertainty)",
}


def compute_uncertainty(components: Mapping[str, float | None], *, spec: UncertaintySpec) -> UncertaintyResult:
    """Mirrors `calibration.confidence.compute_confidence`'s shape:
    `components` maps every NAME in `spec.components` to its value (or
    `None` if genuinely unavailable for this prediction) -- an
    unavailable component is excluded from the aggregate, never coerced
    to `0.0` (Section 16: "Do not silently replace missing components
    with zero")."""
    missing_keys = set(spec.components) - set(components)
    if missing_keys:
        raise UncertaintyPolicyError(f"compute_uncertainty: components mapping is missing declared key(s): {sorted(missing_keys)}")
    availability = {name: (components[name] is not None) for name in spec.components}
    available: dict[str, float] = {}
    for name in spec.components:
        value = components[name]
        if value is None:
            continue
        if not (0.0 <= value <= 1.0):
            raise UncertaintyPolicyError(f"compute_uncertainty: components[{name!r}] must be in [0, 1], got {value!r}")
        available[name] = value
    if not available:
        raise UncertaintyPolicyError("compute_uncertainty: every declared component is unavailable for this prediction")

    values = list(available.values())
    total = statistics.fmean(values) if spec.aggregation == "mean" else max(values)
    unavailable = sorted(set(spec.components) - set(available))
    reason_codes = tuple([f"aggregation:{spec.aggregation}"] + [f"component_unavailable:{name}" for name in unavailable])
    interpretations = "; ".join(f"{name}: {_INTERPRETATIONS.get(name, 'undocumented component')}" for name in sorted(available))

    return UncertaintyResult(
        total_uncertainty=max(0.0, min(1.0, float(total))), components=dict(available), component_availability=availability,
        reason_codes=reason_codes, provenance=(ScoreProvenance.HEURISTIC if len(available) == 1 else ScoreProvenance.COMPOSITE),
        interpretation=interpretations,
    )


__all__ = [
    "UNCERTAINTY_RESULT_SCHEMA_VERSION",
    "UncertaintyResult",
    "bin_support_uncertainty_component",
    "calibrator_disagreement_component",
    "compute_uncertainty",
    "entropy_component",
    "margin_component",
    "model_disagreement_component",
]
