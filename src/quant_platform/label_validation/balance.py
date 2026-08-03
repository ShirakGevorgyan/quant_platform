"""`LabelBalance` (Milestone 11, Phase 3, Part C): binary balance,
multi-class balance, neutral percentage, barrier balance, volatility
bucket balance, extreme imbalance. **No recommendations -- only
evidence**: every `LabelEvidence` this module produces has
`recommendation=None` by construction (`_balance_evidence` never
accepts one) -- a balance FACT ("73% of rows are NEUTRAL") is not a
prescription ("rebalance via class weighting"), which would presume a
specific downstream modeling technique this package has no opinion on.
Also never blocking -- an imbalanced label is not, by itself, grounds
to REJECT a label (unlike degeneracy); it is information a researcher
needs, reported at WARNING/CRITICAL severity as appropriate."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.historical.quality import Severity
from quant_platform.label_validation.distribution import bucket_values, format_discrete_value
from quant_platform.label_validation.evidence import (
    LabelEvidence,
    LabelValidationDimensionKind,
    make_evidence,
)
from quant_platform.labels.builder import LabelBundle
from quant_platform.labels.models import LabelFamily
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = ["EXTREME_IMBALANCE_RATIO_THRESHOLD", "LABEL_BALANCE_SCHEMA_VERSION", "LabelBalance", "compute_label_balance"]

LABEL_BALANCE_SCHEMA_VERSION = 1
EXTREME_IMBALANCE_RATIO_THRESHOLD = 20.0
"""majority_fraction / minority_fraction beyond which balance is
reported as "extreme" -- a documented, disclosed threshold, not a
claim of statistical significance."""

_NEUTRAL_BUCKET_LABEL = format_discrete_value(0.0)


def _balance_evidence(
    *, finding: str, evidence: tuple[str, ...], severity: Severity, affected_labels: tuple[str, ...], statistics: dict[str, float],
) -> LabelEvidence:
    return make_evidence(
        finding=finding, evidence=evidence, dimension=LabelValidationDimensionKind.BALANCE, severity=severity,
        affected_labels=affected_labels, statistics=statistics, recommendation=None, blocking=False, blocking_code=None,
    )


@dataclass(frozen=True, slots=True)
class LabelBalance:
    schema_version: int
    label_specification_id: str
    class_fractions: dict[str, float]
    neutral_fraction: float | None
    """Only populated for `LabelFamily.DIRECTION` (the `0` / NEUTRAL bucket's fraction); `None` for every other family."""
    is_binary: bool
    is_multiclass: bool
    imbalance_ratio: float | None
    """`max class fraction / min class fraction`; `None` with fewer than 2 observed classes."""
    extreme_imbalance: bool
    evidence: tuple[LabelEvidence, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id,
            "class_fractions": self.class_fractions, "neutral_fraction": self.neutral_fraction, "is_binary": self.is_binary,
            "is_multiclass": self.is_multiclass, "imbalance_ratio": self.imbalance_ratio, "extreme_imbalance": self.extreme_imbalance,
            "evidence": [e.to_json_dict() for e in self.evidence],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelBalance:
        require_schema_version(raw, supported=LABEL_BALANCE_SCHEMA_VERSION, context="LabelBalance")
        neutral_raw = raw.get("neutral_fraction")
        imbalance_raw = raw.get("imbalance_ratio")
        return cls(
            schema_version=LABEL_BALANCE_SCHEMA_VERSION, label_specification_id=str(raw["label_specification_id"]),
            class_fractions={str(k): float(str(v)) for k, v in as_json_dict(raw.get("class_fractions") or {}, field_name="class_fractions").items()},
            neutral_fraction=(None if neutral_raw is None else float(str(neutral_raw))), is_binary=bool(raw["is_binary"]),
            is_multiclass=bool(raw["is_multiclass"]), imbalance_ratio=(None if imbalance_raw is None else float(str(imbalance_raw))),
            extreme_imbalance=bool(raw["extreme_imbalance"]),
            evidence=tuple(
                LabelEvidence.from_json_dict(as_json_dict(e, field_name="evidence[]"))
                for e in as_json_list(raw.get("evidence") or [], field_name="evidence")
            ),
        )


def compute_label_balance(bundle: LabelBundle, *, bucket_count: int = 10) -> LabelBalance:
    spec_id = bundle.specification.label_specification_id
    buckets = bucket_values(bundle.values, label_family=bundle.specification.label_family, bucket_count=bucket_count)
    valid_buckets = buckets.dropna()
    evidence: list[LabelEvidence] = []

    if len(valid_buckets) == 0:
        return LabelBalance(
            schema_version=LABEL_BALANCE_SCHEMA_VERSION, label_specification_id=spec_id, class_fractions={}, neutral_fraction=None,
            is_binary=False, is_multiclass=False, imbalance_ratio=None, extreme_imbalance=False, evidence=(),
        )

    counts = valid_buckets.value_counts()
    fractions = counts / len(valid_buckets)
    class_fractions = {str(k): float(v) for k, v in fractions.items()}
    class_count = len(class_fractions)
    is_binary = class_count == 2
    is_multiclass = class_count > 2

    neutral_fraction = None
    if bundle.specification.label_family is LabelFamily.DIRECTION:
        neutral_fraction = class_fractions.get(_NEUTRAL_BUCKET_LABEL, 0.0)
        evidence.append(_balance_evidence(
            finding=f"neutral fraction is {neutral_fraction:.4f}", evidence=(f"class_fractions={class_fractions}",),
            severity=(Severity.WARNING if neutral_fraction > 0.9 else Severity.INFO), affected_labels=(spec_id,),
            statistics={"neutral_fraction": neutral_fraction},
        ))

    imbalance_ratio = None
    extreme_imbalance = False
    if class_count >= 2:
        imbalance_ratio = max(class_fractions.values()) / min(class_fractions.values())
        extreme_imbalance = imbalance_ratio > EXTREME_IMBALANCE_RATIO_THRESHOLD
        evidence.append(_balance_evidence(
            finding=f"class imbalance ratio is {imbalance_ratio:.4f}" + (" (extreme)" if extreme_imbalance else ""),
            evidence=(f"class_fractions={class_fractions}",), severity=(Severity.WARNING if extreme_imbalance else Severity.INFO),
            affected_labels=(spec_id,), statistics={"imbalance_ratio": imbalance_ratio},
        ))
    else:
        evidence.append(_balance_evidence(
            finding=f"only {class_count} class observed", evidence=(f"class_fractions={class_fractions}",), severity=Severity.WARNING,
            affected_labels=(spec_id,), statistics={"class_count": float(class_count)},
        ))

    return LabelBalance(
        schema_version=LABEL_BALANCE_SCHEMA_VERSION, label_specification_id=spec_id, class_fractions=class_fractions,
        neutral_fraction=neutral_fraction, is_binary=is_binary, is_multiclass=is_multiclass, imbalance_ratio=imbalance_ratio,
        extreme_imbalance=extreme_imbalance, evidence=tuple(evidence),
    )
