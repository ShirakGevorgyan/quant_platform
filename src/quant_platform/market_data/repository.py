"""Repository facade (Milestone 10, Phase 2) -- bundles every durable
store one repository root needs: Phase 1's `MarketEventStore`/
`FeatureStore` (primary, append-only economic facts) plus Phase 2's
`DatasetManifestStore`/`PartitionStore` (derived, versioned/rebuildable
index structures over those facts). Every other Phase 2 module
(`ingestion.py`, `checkpoints.py`, `recovery.py`, `reconciliation.py`,
`compaction.py`, `export.py`) takes a `MarketDataRepository` rather than
four separate store objects, so a caller opens exactly one root once."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_platform.market_data.events import MarketEventStore
from quant_platform.market_data.feature_store import FeatureStore
from quant_platform.market_data.manifests import DatasetManifestStore
from quant_platform.market_data.partitions import PartitionStore

__all__ = ["MarketDataRepository"]


@dataclass(frozen=True, slots=True)
class MarketDataRepository:
    root: Path
    event_store: MarketEventStore
    feature_store: FeatureStore
    manifest_store: DatasetManifestStore
    partition_store: PartitionStore

    @classmethod
    def open(cls, storage_root: Path | str) -> MarketDataRepository:
        root = Path(storage_root).resolve()
        return cls(
            root=root, event_store=MarketEventStore(root), feature_store=FeatureStore(root),
            manifest_store=DatasetManifestStore(root), partition_store=PartitionStore(root),
        )
