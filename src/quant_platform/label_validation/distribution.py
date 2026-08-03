"""`LabelDistribution` (Milestone 11, Phase 3, Part C): entropy,
cardinality, effective cardinality, class ratios, tail ratios, rare
labels, sparsity -- the pure SHAPE of a label's own value distribution,
never its relationship to any prediction target.

Also owns the shared bucketing helper `bucket_values`/`is_discrete_family`
every other dimension in this package (`balance.py`, `degeneracy.py`)
reuses: `DIRECTION`/`TRIPLE_BARRIER` are naturally discrete (a handful
of exact values) and are grouped by their exact value; every other
family is CONTINUOUS and is grouped into quantile buckets (deciles by
default) -- this is what makes "volatility bucket balance" meaningful
for `FORWARD_VOLATILITY` (and the analogous "return bucket balance" for
`NEXT_RETURN`/`MULTI_HORIZON_RETURN`) rather than treating every raw
float as its own singleton class."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.historical.quality import Severity
from quant_platform.label_validation.evidence import (
    LabelEvidence,
    LabelValidationDimensionKind,
    make_evidence,
)
from quant_platform.labels.builder import LabelBundle
from quant_platform.labels.models import LabelFamily
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = [
    "DISCRETE_LABEL_FAMILIES",
    "LABEL_DISTRIBUTION_SCHEMA_VERSION",
    "LabelDistribution",
    "bucket_values",
    "compute_label_distribution",
    "format_discrete_value",
    "is_discrete_family",
]

LABEL_DISTRIBUTION_SCHEMA_VERSION = 1

DISCRETE_LABEL_FAMILIES = frozenset({LabelFamily.DIRECTION, LabelFamily.TRIPLE_BARRIER})


def is_discrete_family(label_family: LabelFamily) -> bool:
    return label_family in DISCRETE_LABEL_FAMILIES


def format_discrete_value(value: float) -> str:
    return f"{value:g}"


def bucket_values(values: pd.Series, *, label_family: LabelFamily, bucket_count: int = 10) -> pd.Series:
    """A series of bucket-label strings (or `None` for a missing input),
    aligned 1:1 with `values`. Discrete families group by exact value;
    continuous families group by quantile bucket (`pandas.qcut`,
    `duplicates="drop"` -- a low-cardinality or near-constant continuous
    series gracefully collapses to fewer buckets rather than raising)."""
    if is_discrete_family(label_family):
        return values.map(lambda v: None if pd.isna(v) else format_discrete_value(v))

    valid = values.dropna()
    if valid.empty:
        return pd.Series([None] * len(values), index=values.index, dtype=object)
    if valid.nunique() == 1:
        single = format_discrete_value(float(valid.iloc[0]))
        return values.map(lambda v: single if pd.notna(v) else None)

    try:
        bucketed = pd.qcut(values, q=bucket_count, duplicates="drop")
    except ValueError:
        bucketed = pd.qcut(values, q=1, duplicates="drop")
    return bucketed.astype(object).map(lambda v: None if pd.isna(v) else str(v))


@dataclass(frozen=True, slots=True)
class LabelDistribution:
    schema_version: int
    label_specification_id: str
    cardinality: int
    effective_cardinality: float
    """`2 ** entropy` (base-2 entropy) -- the "effective number of
    categories" a uniform distribution with the same entropy would have."""
    entropy: float
    class_ratios: dict[str, float]
    tail_ratios: dict[str, float]
    rare_labels: tuple[str, ...]
    sparsity: float
    """`missing_count / row_count`."""
    evidence: tuple[LabelEvidence, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id, "cardinality": self.cardinality,
            "effective_cardinality": self.effective_cardinality, "entropy": self.entropy, "class_ratios": self.class_ratios,
            "tail_ratios": self.tail_ratios, "rare_labels": list(self.rare_labels), "sparsity": self.sparsity,
            "evidence": [e.to_json_dict() for e in self.evidence],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelDistribution:
        require_schema_version(raw, supported=LABEL_DISTRIBUTION_SCHEMA_VERSION, context="LabelDistribution")
        return cls(
            schema_version=LABEL_DISTRIBUTION_SCHEMA_VERSION, label_specification_id=str(raw["label_specification_id"]),
            cardinality=int(str(raw["cardinality"])), effective_cardinality=float(str(raw["effective_cardinality"])),
            entropy=float(str(raw["entropy"])),
            class_ratios={str(k): float(str(v)) for k, v in as_json_dict(raw.get("class_ratios") or {}, field_name="class_ratios").items()},
            tail_ratios={str(k): float(str(v)) for k, v in as_json_dict(raw.get("tail_ratios") or {}, field_name="tail_ratios").items()},
            rare_labels=tuple(str(s) for s in as_json_list(raw.get("rare_labels") or [], field_name="rare_labels")), sparsity=float(str(raw["sparsity"])),
            evidence=tuple(
                LabelEvidence.from_json_dict(as_json_dict(e, field_name="evidence[]"))
                for e in as_json_list(raw.get("evidence") or [], field_name="evidence")
            ),
        )


def compute_label_distribution(bundle: LabelBundle, *, bucket_count: int = 10, rare_threshold: float = 0.01) -> LabelDistribution:
    row_count = bundle.row_count
    buckets = bucket_values(bundle.values, label_family=bundle.specification.label_family, bucket_count=bucket_count)
    valid_buckets = buckets.dropna()
    valid_count = len(valid_buckets)
    sparsity = (row_count - valid_count) / row_count if row_count else 0.0
    spec_id = bundle.specification.label_specification_id

    if valid_count == 0:
        return LabelDistribution(
            schema_version=LABEL_DISTRIBUTION_SCHEMA_VERSION, label_specification_id=spec_id, cardinality=0, effective_cardinality=0.0,
            entropy=0.0, class_ratios={}, tail_ratios={}, rare_labels=(), sparsity=sparsity, evidence=(),
        )

    counts = valid_buckets.value_counts()
    fractions = counts / valid_count
    class_ratios = {str(k): float(v) for k, v in fractions.items()}
    probabilities = fractions.to_numpy()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    effective_cardinality = float(2.0**entropy)
    cardinality = int(counts.shape[0])
    rare_labels = tuple(sorted(k for k, v in class_ratios.items() if v < rare_threshold))
    sorted_fractions = sorted(class_ratios.values())
    tail_ratios = {"smallest_class_fraction": sorted_fractions[0], "largest_class_fraction": sorted_fractions[-1]}

    evidence: list[LabelEvidence] = []
    if rare_labels:
        evidence.append(make_evidence(
            finding=f"{len(rare_labels)} rare label class(es) observed (below {rare_threshold:.2%} of valid rows)",
            evidence=(f"rare_labels={rare_labels}",), dimension=LabelValidationDimensionKind.DISTRIBUTION, severity=Severity.INFO,
            affected_labels=(spec_id,), statistics={"rare_label_count": float(len(rare_labels))},
        ))
    if sparsity > 0.5:
        evidence.append(make_evidence(
            finding=f"sparsity is high ({sparsity:.4f} of rows are missing)", evidence=(f"valid_count={valid_count}", f"row_count={row_count}"),
            dimension=LabelValidationDimensionKind.DISTRIBUTION, severity=Severity.WARNING, affected_labels=(spec_id,),
            statistics={"sparsity": sparsity},
        ))

    return LabelDistribution(
        schema_version=LABEL_DISTRIBUTION_SCHEMA_VERSION, label_specification_id=spec_id, cardinality=cardinality,
        effective_cardinality=effective_cardinality, entropy=entropy, class_ratios=class_ratios, tail_ratios=tail_ratios,
        rare_labels=rare_labels, sparsity=sparsity, evidence=tuple(evidence),
    )
