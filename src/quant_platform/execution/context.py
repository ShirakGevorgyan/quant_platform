"""Immutable per-fold execution context (Milestone 4B, Section 6) --
everything a `FoldExecutor` needs to run exactly one fold, bundled into
one frozen object so an executor implementation never has to reach back
into the runner's own mutable state. "Immutable" describes this object's
OWN fields (never reassigned once built, one fresh context per fold);
like `ml.registry.ModelDefinition` holding a `ModelFactory`, it may still
reference already-constructed, longer-lived collaborator objects (the
artifact/event stores) that are themselves stateful services, not pure
data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.manifests import ExperimentManifest
from quant_platform.ml.models import EnvironmentSnapshot
from quant_platform.ml.persistence import parse_utc_timestamp
from quant_platform.ml.tracking import ExperimentEventStore


@dataclass(frozen=True, slots=True)
class FoldExecutionContext:
    experiment_id: str
    fold_index: int
    split_id: str
    dataset_content_id: str
    manifest: ExperimentManifest
    seed: int
    environment: EnvironmentSnapshot
    artifact_store: MLArtifactStore
    event_store: ExperimentEventStore
    artifacts_root: Path
    started_at: str

    def __post_init__(self) -> None:
        if self.fold_index < 0:
            raise ValueError(f"FoldExecutionContext.fold_index must be >= 0, got {self.fold_index}")
        if not self.split_id:
            raise ValueError("FoldExecutionContext.split_id must not be empty")
        if not self.dataset_content_id:
            raise ValueError("FoldExecutionContext.dataset_content_id must not be empty")
        if self.seed < 0:
            raise ValueError(f"FoldExecutionContext.seed must be >= 0, got {self.seed}")
        parse_utc_timestamp(self.started_at)


__all__ = ["FoldExecutionContext"]
