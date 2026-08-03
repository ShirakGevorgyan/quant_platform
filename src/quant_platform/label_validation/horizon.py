"""Horizon analysis (Milestone 11, Phase 3, Part C): compares
coverage/balance/degeneracy/stability across a set of same-family,
different-horizon `labels.builder.LabelBundle`s (e.g. the 6 minimum
required Multi Horizon Return horizons: 1, 5, 10, 20, 50, 100). **Only
descriptive statistics** -- `HorizonComparisonReport` never ranks or
scores which horizon is "best"; it lists each horizon's own facts side
by side, reusing `coverage.py`/`degeneracy.py`/`balance.py`/
`stability.py`'s own compute functions directly rather than
duplicating any of their logic."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.core.exceptions import LabelValidationRequestError
from quant_platform.label_validation.balance import compute_label_balance
from quant_platform.label_validation.coverage import compute_label_coverage
from quant_platform.label_validation.degeneracy import compute_label_degeneracy
from quant_platform.label_validation.stability import compute_label_stability
from quant_platform.labels.builder import LabelBundle
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = ["HORIZON_ANALYSIS_SCHEMA_VERSION", "HorizonComparisonReport", "HorizonSummary", "compare_horizons"]

HORIZON_ANALYSIS_SCHEMA_VERSION = 1


def _horizon_bars_of(bundle: LabelBundle) -> int:
    parameters = bundle.specification.parameters
    for key in ("horizon_bars", "max_holding_bars"):
        if key in parameters:
            return int(str(parameters[key]))
    raise LabelValidationRequestError(
        f"specification {bundle.specification.label_specification_id!r} carries neither 'horizon_bars' nor 'max_holding_bars' -- "
        "cannot compare horizons without a horizon parameter",
        context={"label_specification_id": bundle.specification.label_specification_id},
    )


@dataclass(frozen=True, slots=True)
class HorizonSummary:
    horizon_bars: int
    label_specification_id: str
    coverage_fraction: float
    is_degenerate: bool
    imbalance_ratio: float | None
    window_stability_score: float | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "horizon_bars": self.horizon_bars, "label_specification_id": self.label_specification_id,
            "coverage_fraction": self.coverage_fraction, "is_degenerate": self.is_degenerate, "imbalance_ratio": self.imbalance_ratio,
            "window_stability_score": self.window_stability_score,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> HorizonSummary:
        imbalance_raw = raw.get("imbalance_ratio")
        stability_raw = raw.get("window_stability_score")
        return cls(
            horizon_bars=int(str(raw["horizon_bars"])), label_specification_id=str(raw["label_specification_id"]),
            coverage_fraction=float(str(raw["coverage_fraction"])), is_degenerate=bool(raw["is_degenerate"]),
            imbalance_ratio=(None if imbalance_raw is None else float(str(imbalance_raw))),
            window_stability_score=(None if stability_raw is None else float(str(stability_raw))),
        )


@dataclass(frozen=True, slots=True)
class HorizonComparisonReport:
    schema_version: int
    label_family: str
    horizons: tuple[HorizonSummary, ...]
    """Sorted by `horizon_bars` ascending."""

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "label_family": self.label_family, "horizons": [h.to_json_dict() for h in self.horizons]}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> HorizonComparisonReport:
        require_schema_version(raw, supported=HORIZON_ANALYSIS_SCHEMA_VERSION, context="HorizonComparisonReport")
        return cls(
            schema_version=HORIZON_ANALYSIS_SCHEMA_VERSION, label_family=str(raw["label_family"]),
            horizons=tuple(
                HorizonSummary.from_json_dict(as_json_dict(h, field_name="horizons[]"))
                for h in as_json_list(raw.get("horizons") or [], field_name="horizons")
            ),
        )


def compare_horizons(bundles: tuple[LabelBundle, ...]) -> HorizonComparisonReport:
    if not bundles:
        raise LabelValidationRequestError("compare_horizons requires at least one bundle", context={})
    family = bundles[0].specification.label_family
    if any(b.specification.label_family != family for b in bundles):
        raise LabelValidationRequestError(
            "compare_horizons requires every bundle to share the same label_family", context={"label_family": family.value},
        )

    summaries = []
    for bundle in bundles:
        coverage = compute_label_coverage(bundle)
        degeneracy = compute_label_degeneracy(bundle)
        balance = compute_label_balance(bundle)
        stability = compute_label_stability(bundle)
        summaries.append(HorizonSummary(
            horizon_bars=_horizon_bars_of(bundle), label_specification_id=bundle.specification.label_specification_id,
            coverage_fraction=coverage.coverage_fraction, is_degenerate=(degeneracy.is_constant or degeneracy.is_empty),
            imbalance_ratio=balance.imbalance_ratio, window_stability_score=stability.temporal.window_stability_score,
        ))
    summaries.sort(key=lambda s: s.horizon_bars)

    return HorizonComparisonReport(schema_version=HORIZON_ANALYSIS_SCHEMA_VERSION, label_family=family.value, horizons=tuple(summaries))
