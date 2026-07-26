"""Feature lineage: a traceable, human-readable record of exactly how one
computed feature column came to exist -- source dataset, source
symbol/timeframe, raw input columns, the transformation applied, its
parameters, any feature-on-feature dependencies, the lookback/availability
contract, and a code fingerprint. Every feature a `dataset_builder` writes
into a research dataset carries one of these, embedded in the dataset
manifest (`manifests.ResearchDatasetManifest.feature_lineages`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from quant_platform.core.exceptions import FeatureError
from quant_platform.features.models import FeatureSpec


def _as_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise FeatureError(f"Expected a JSON array, got {type(value).__name__}")
    return value


def _as_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FeatureError(f"Expected a JSON object, got {type(value).__name__}")
    return value


@dataclass(frozen=True, slots=True)
class FeatureLineage:
    feature_name: str
    feature_version: str
    category: str
    source_dataset_manifest_id: str | None
    source_symbols: tuple[str, ...]
    source_timeframe: str
    required_inputs: tuple[str, ...]
    transformation: str
    parameters: dict[str, object] = field(default_factory=dict)
    feature_dependencies: tuple[str, ...] = ()
    lookback_bars: int = 0
    warmup_bars: int = 0
    availability_delay_seconds: float = 0.0
    spec_fingerprint: str = ""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "feature_name": self.feature_name,
            "feature_version": self.feature_version,
            "category": self.category,
            "source_dataset_manifest_id": self.source_dataset_manifest_id,
            "source_symbols": list(self.source_symbols),
            "source_timeframe": self.source_timeframe,
            "required_inputs": list(self.required_inputs),
            "transformation": self.transformation,
            "parameters": self.parameters,
            "feature_dependencies": list(self.feature_dependencies),
            "lookback_bars": self.lookback_bars,
            "warmup_bars": self.warmup_bars,
            "availability_delay_seconds": self.availability_delay_seconds,
            "spec_fingerprint": self.spec_fingerprint,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureLineage:
        return cls(
            feature_name=str(raw["feature_name"]),
            feature_version=str(raw["feature_version"]),
            category=str(raw["category"]),
            source_dataset_manifest_id=(
                None if raw.get("source_dataset_manifest_id") is None else str(raw["source_dataset_manifest_id"])
            ),
            source_symbols=tuple(str(s) for s in _as_list(raw["source_symbols"])),
            source_timeframe=str(raw["source_timeframe"]),
            required_inputs=tuple(str(s) for s in _as_list(raw["required_inputs"])),
            transformation=str(raw["transformation"]),
            parameters=_as_dict(raw.get("parameters") or {}),
            feature_dependencies=tuple(str(s) for s in _as_list(raw.get("feature_dependencies") or [])),
            lookback_bars=int(str(raw.get("lookback_bars", 0))),
            warmup_bars=int(str(raw.get("warmup_bars", 0))),
            availability_delay_seconds=float(str(raw.get("availability_delay_seconds", 0.0))),
            spec_fingerprint=str(raw.get("spec_fingerprint", "")),
        )


def build_lineage(
    spec: FeatureSpec, *, source_dataset_manifest_id: str | None, transformation: str
) -> FeatureLineage:
    """Construct a `FeatureLineage` from a feature's own `FeatureSpec` plus
    the one piece of context a spec cannot know about itself: which exact
    historical dataset manifest it was computed against."""
    return FeatureLineage(
        feature_name=spec.name,
        feature_version=spec.version,
        category=spec.category.value,
        source_dataset_manifest_id=source_dataset_manifest_id,
        source_symbols=spec.source_symbols,
        source_timeframe=spec.source_timeframe.value,
        required_inputs=spec.required_inputs,
        transformation=transformation,
        parameters=dict(spec.deterministic_params),
        feature_dependencies=spec.feature_dependencies,
        lookback_bars=spec.lookback_bars,
        warmup_bars=spec.warmup_bars,
        availability_delay_seconds=spec.availability_delay.total_seconds(),
        spec_fingerprint=spec.fingerprint(),
    )


def render_lineage_report(lineages: Sequence[FeatureLineage]) -> str:
    """A plain-text, human-readable lineage report -- one block per
    feature, sorted by name for a stable diff-friendly rendering."""
    lines: list[str] = [f"Feature lineage report ({len(lineages)} feature(s))", "=" * 60]
    for lineage in sorted(lineages, key=lambda ln: ln.feature_name):
        lines.append(f"\n{lineage.feature_name} (v{lineage.feature_version})  [{lineage.category}]")
        lines.append(f"  source: symbols={list(lineage.source_symbols)} timeframe={lineage.source_timeframe}")
        lines.append(f"  dataset manifest: {lineage.source_dataset_manifest_id}")
        lines.append(f"  required inputs: {list(lineage.required_inputs)}")
        lines.append(f"  transformation: {lineage.transformation}")
        if lineage.parameters:
            lines.append(f"  parameters: {lineage.parameters}")
        if lineage.feature_dependencies:
            lines.append(f"  depends on features: {list(lineage.feature_dependencies)}")
        lines.append(f"  lookback_bars={lineage.lookback_bars} warmup_bars={lineage.warmup_bars}")
        if lineage.availability_delay_seconds:
            lines.append(f"  availability_delay={lineage.availability_delay_seconds}s")
        lines.append(f"  spec_fingerprint: {lineage.spec_fingerprint[:16]}...")
    return "\n".join(lines)


__all__ = ["FeatureLineage", "build_lineage", "render_lineage_report"]
