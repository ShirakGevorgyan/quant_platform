"""`LabelStatistics` (Milestone 11, Phase 3, Part C): the foundational
descriptive-statistics summary every other dimension in this package
(`distribution.py`, `balance.py`, `degeneracy.py`, `coverage.py`)
computes from -- basic numeric facts about a `labels.builder.LabelBundle`'s
own `values`, computed exactly ONCE per qualification run, never a
predictive or target-dependent statistic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quant_platform.labels.builder import LabelBundle
from quant_platform.ml.persistence import require_schema_version

__all__ = ["LABEL_STATISTICS_SCHEMA_VERSION", "LabelStatistics", "compute_label_statistics"]

LABEL_STATISTICS_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LabelStatistics:
    schema_version: int
    label_specification_id: str
    row_count: int
    valid_count: int
    missing_count: int
    mean: float | None
    std: float | None
    minimum: float | None
    maximum: float | None
    median: float | None
    p05: float | None
    p95: float | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id, "row_count": self.row_count,
            "valid_count": self.valid_count, "missing_count": self.missing_count, "mean": self.mean, "std": self.std,
            "minimum": self.minimum, "maximum": self.maximum, "median": self.median, "p05": self.p05, "p95": self.p95,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelStatistics:
        require_schema_version(raw, supported=LABEL_STATISTICS_SCHEMA_VERSION, context="LabelStatistics")

        def _optional_float(key: str) -> float | None:
            value = raw.get(key)
            return None if value is None else float(str(value))

        return cls(
            schema_version=LABEL_STATISTICS_SCHEMA_VERSION, label_specification_id=str(raw["label_specification_id"]),
            row_count=int(str(raw["row_count"])), valid_count=int(str(raw["valid_count"])), missing_count=int(str(raw["missing_count"])),
            mean=_optional_float("mean"), std=_optional_float("std"), minimum=_optional_float("minimum"), maximum=_optional_float("maximum"),
            median=_optional_float("median"), p05=_optional_float("p05"), p95=_optional_float("p95"),
        )


def compute_label_statistics(bundle: LabelBundle) -> LabelStatistics:
    valid = bundle.values.dropna()
    row_count = bundle.row_count
    valid_count = len(valid)
    missing_count = row_count - valid_count

    if valid_count == 0:
        return LabelStatistics(
            schema_version=LABEL_STATISTICS_SCHEMA_VERSION, label_specification_id=bundle.specification.label_specification_id,
            row_count=row_count, valid_count=0, missing_count=missing_count, mean=None, std=None, minimum=None, maximum=None,
            median=None, p05=None, p95=None,
        )

    std = float(valid.std(ddof=0)) if valid_count > 1 else 0.0
    return LabelStatistics(
        schema_version=LABEL_STATISTICS_SCHEMA_VERSION, label_specification_id=bundle.specification.label_specification_id,
        row_count=row_count, valid_count=valid_count, missing_count=missing_count, mean=float(valid.mean()), std=std,
        minimum=float(valid.min()), maximum=float(valid.max()), median=float(valid.median()),
        p05=float(np.percentile(valid.to_numpy(), 5)), p95=float(np.percentile(valid.to_numpy(), 95)),
    )
