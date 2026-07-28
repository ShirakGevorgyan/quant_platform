"""Confidence framework (Milestone 4E, Section 15) -- confidence is
DEFINED, not assumed to be "the calibrated probability itself".

COMPONENTS ARE COMPUTED BY THE CALLER, COMBINED HERE
--------------------------------------------------------------------------
`compute_confidence` never reaches into a `RawPredictionSet`/reliability
report/model ensemble itself -- it combines an already-computed mapping
of NAMED components (`distance_from_threshold`/`probability_extremity`,
computed here via small pure helpers; `calibration_bin_support`/
`model_disagreement`/`calibrator_disagreement`, computed by
`calibration.fitting`/`calibration.uncertainty`, which have the
additional context -- a reliability report, multiple inner models --
this module deliberately does not depend on). This keeps "how components
combine into one score" (a pure, always-testable function of numbers)
completely separate from "what each component means" (each documented at
its own point of computation).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from quant_platform.calibration.models import ConfidenceCategory, ScoreProvenance
from quant_platform.calibration.specs import ConfidenceSpec
from quant_platform.core.exceptions import ConfidencePolicyError
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

CONFIDENCE_RESULT_SCHEMA_VERSION = 1
_DEFAULT_COMPONENT = "distance_from_threshold"


def distance_from_threshold_component(probability: float, threshold: float) -> float:
    """Normalized to `[0, 1]`: `0.0` exactly at the decision threshold,
    `1.0` at the probability furthest from it (`0.0` or `1.0`,
    whichever is farther from `threshold`)."""
    if not (0.0 <= probability <= 1.0):
        raise ConfidencePolicyError(f"distance_from_threshold_component: probability must be in [0, 1], got {probability!r}")
    if not (0.0 <= threshold <= 1.0):
        raise ConfidencePolicyError(f"distance_from_threshold_component: threshold must be in [0, 1], got {threshold!r}")
    normalizer = max(threshold, 1.0 - threshold)
    if normalizer == 0.0:
        return 0.0
    return min(abs(probability - threshold) / normalizer, 1.0)


def probability_extremity_component(probability: float) -> float:
    """`0.0` at `probability == 0.5` (maximally uncertain), `1.0` at
    `probability in {0.0, 1.0}` (maximally extreme)."""
    if not (0.0 <= probability <= 1.0):
        raise ConfidencePolicyError(f"probability_extremity_component: probability must be in [0, 1], got {probability!r}")
    return abs(2.0 * probability - 1.0)


@dataclass(frozen=True, slots=True)
class ConfidenceResult:
    score: float
    category: str
    components: Mapping[str, float]
    component_availability: Mapping[str, bool]
    reason_codes: tuple[str, ...]
    provenance: ScoreProvenance

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ConfidencePolicyError(f"ConfidenceResult.score must be in [0, 1], got {self.score!r}")
        if not math.isfinite(self.score):
            raise ConfidencePolicyError(f"ConfidenceResult.score must be finite, got {self.score!r}")
        try:
            ConfidenceCategory(self.category)
        except ValueError as exc:
            raise ConfidencePolicyError(f"ConfidenceResult.category {self.category!r} is not a known ConfidenceCategory") from exc
        for name, value in self.components.items():
            if not math.isfinite(value):
                raise ConfidencePolicyError(f"ConfidenceResult.components[{name!r}] must be finite, got {value!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": CONFIDENCE_RESULT_SCHEMA_VERSION, "score": self.score, "category": self.category,
            "components": dict(sorted(self.components.items())), "component_availability": dict(sorted(self.component_availability.items())),
            "reason_codes": list(self.reason_codes), "provenance": self.provenance.value,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ConfidenceResult:
        require_schema_version(raw, supported=CONFIDENCE_RESULT_SCHEMA_VERSION, context="ConfidenceResult")
        return cls(
            score=float(str(raw["score"])), category=str(raw["category"]),
            components={str(k): float(v) for k, v in as_json_dict(raw["components"], field_name="components").items()},
            component_availability={str(k): bool(v) for k, v in as_json_dict(raw["component_availability"], field_name="component_availability").items()},
            reason_codes=tuple(str(c) for c in as_json_list(raw["reason_codes"], field_name="reason_codes")),
            provenance=ScoreProvenance(raw["provenance"]),
        )


def compute_confidence(components: Mapping[str, float | None], *, spec: ConfidenceSpec) -> ConfidenceResult:
    """`components` maps a component name to its value in `[0, 1]`, or
    `None` if that component is genuinely unavailable for this
    prediction (Section 15/17: "Do not silently replace missing
    components with zero" -- an unavailable component is simply excluded
    from the weighted combination, never coerced to `0.0`, and its
    unavailability is recorded in `component_availability`).

    If `spec.component_weights` is empty, this is a single-component
    score: `components["distance_from_threshold"]` alone, and the
    result's `provenance` is `HEURISTIC`. Otherwise every KEY in
    `component_weights` must be present in `components` (even if its
    value is `None`); the combination is a weighted average over the
    AVAILABLE ones only, weights renormalized to those -- and
    `provenance` is `COMPOSITE`."""
    availability = {name: (value is not None) for name, value in components.items()}

    if not spec.component_weights:
        if _DEFAULT_COMPONENT not in components or components[_DEFAULT_COMPONENT] is None:
            raise ConfidencePolicyError(
                f"compute_confidence: component_weights is empty (single-component mode) but "
                f"components[{_DEFAULT_COMPONENT!r}] is missing or None"
            )
        score = components[_DEFAULT_COMPONENT]
        assert score is not None
        used = {_DEFAULT_COMPONENT: score}
        provenance = ScoreProvenance.HEURISTIC
        reason_codes: tuple[str, ...] = ("single_component_distance_from_threshold",)
    else:
        missing_keys = set(spec.component_weights) - set(components)
        if missing_keys:
            raise ConfidencePolicyError(f"compute_confidence: components mapping is missing declared key(s): {sorted(missing_keys)}")
        available_weight = sum(w for name, w in spec.component_weights.items() if components[name] is not None)
        if available_weight <= 0.0:
            raise ConfidencePolicyError(
                "compute_confidence: every weighted component is unavailable for this prediction -- cannot "
                "compute a composite confidence score"
            )
        used = {}
        score = 0.0
        for name, weight in spec.component_weights.items():
            value = components[name]
            if value is None:
                continue
            if not (0.0 <= value <= 1.0):
                raise ConfidencePolicyError(f"compute_confidence: components[{name!r}] must be in [0, 1], got {value!r}")
            used[name] = value
            score += (weight / available_weight) * value
        provenance = ScoreProvenance.COMPOSITE
        unavailable = sorted(name for name in spec.component_weights if components[name] is None)
        reason_codes = tuple(["composite_weighted_average"] + [f"component_unavailable:{name}" for name in unavailable])

    score = max(0.0, min(1.0, score))
    return ConfidenceResult(
        score=score, category=spec.category_for(score), components=used, component_availability=availability,
        reason_codes=reason_codes, provenance=provenance,
    )


__all__ = [
    "CONFIDENCE_RESULT_SCHEMA_VERSION",
    "ConfidenceResult",
    "compute_confidence",
    "distance_from_threshold_component",
    "probability_extremity_component",
]
