"""Safe physical compaction (Milestone 10, Phase 2, OPTIONAL per
specification -- "only if it can be done safely within scope").

SCOPE DECISION: compaction here means REBUILDING every partition and the
manifest for a dataset fresh from its own current, already-durable raw/
feature data -- normalizing the physical storage representation back to
exactly what `build_partition`/`rebuild_dataset_manifest_from_events`/
`rebuild_feature_dataset_manifest` would produce from scratch (the same
operation `recovery.py` performs after an interruption, here invoked as
routine maintenance rather than crash recovery). This is the SAFE subset
of "compaction" the specification describes: it satisfies "rebuild
indexes" and "normalize canonical storage representation" literally.

DELIBERATELY NOT IMPLEMENTED: combining small partitions into larger ones
by changing PARTITIONING GRANULARITY (e.g. daily -> monthly). Doing that
safely would require partition physical paths to be scoped by
granularity (so old daily and new monthly partition files never collide
at the same path) -- a storage-layout change touching every module that
writes a `Partition` (`ingestion.py`, `feature_generation.py`,
`recovery.py`, `reconciliation.py`). The specification explicitly marks
compaction optional and permits choosing a safely-scoped subset; this is
that scope decision, documented rather than silently narrowed.

INVARIANTS THIS MODULE GUARANTEES (verified by
`tests/unit/market_data/test_market_data_compaction.py`): event/feature
identities never change (nothing about `MarketEventStore`/`FeatureStore`
content is touched); `semantic_digest` never changes (it is recomputed
from the same underlying data every time, compaction or not); logical
ordering never changes (member ordering is always `(member_time,
member_id)`, independent of physical rebuild); replay result never
changes (`replay.py` reads raw events, never partition files, so
compaction is invisible to it). `dataset_id` MAY change if a partition
was physically hand-edited/drifted from its canonical form (compaction
corrects it back), in which case `ordered_partition_ids` differs -- a
NEW, still fully valid manifest VERSION, never a mutation of a prior
one."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quant_platform.core.exceptions import RepositoryCorruptionError
from quant_platform.market_data.events import market_data_event_id, market_data_event_time
from quant_platform.market_data.feature_generation import rebuild_feature_dataset_manifest
from quant_platform.market_data.identity import require_tz_aware
from quant_platform.market_data.ingestion import (
    rebuild_dataset_manifest_from_events,
    rebuild_touched_partitions,
)
from quant_platform.market_data.manifests import DatasetKey, DatasetKind, DatasetManifest, PartitioningSpec
from quant_platform.market_data.partitions import partition_key_for
from quant_platform.market_data.repository import MarketDataRepository

__all__ = ["CompactionResult", "compact_feature_dataset", "compact_raw_dataset"]


@dataclass(frozen=True, slots=True)
class CompactionResult:
    dataset_key: DatasetKey
    manifest_before: DatasetManifest | None
    manifest_after: DatasetManifest
    semantic_digest_preserved: bool
    rebuilt_partition_keys: tuple[str, ...]


def compact_raw_dataset(*, repository: MarketDataRepository, dataset_key: DatasetKey, partitioning: PartitioningSpec, compaction_time: datetime) -> CompactionResult:
    if dataset_key.dataset_kind is not DatasetKind.RAW_MARKET_EVENTS:
        raise RepositoryCorruptionError("compact_raw_dataset requires a RAW_MARKET_EVENTS dataset_key")
    require_tz_aware(compaction_time, field_name="compaction_time")
    assert dataset_key.provider is not None
    manifest_before = repository.manifest_store.read_current(dataset_key)

    events = repository.event_store.read_events(dataset_key.provider, dataset_key.instrument_id)
    all_members = [(market_data_event_id(e), market_data_event_time(e)) for e in events]
    touched_partition_keys = {partition_key_for(t, partitioning) for _, t in all_members}
    if touched_partition_keys:
        rebuild_touched_partitions(repository=repository, dataset_key=dataset_key, partitioning=partitioning, all_members=all_members, touched_partition_keys=touched_partition_keys)

    manifest_after = rebuild_dataset_manifest_from_events(repository=repository, dataset_key=dataset_key, partitioning=partitioning, raw_source_dataset_id=None, creation_time=compaction_time)
    semantic_digest_preserved = manifest_before is None or manifest_before.semantic_digest == manifest_after.semantic_digest
    return CompactionResult(
        dataset_key=dataset_key, manifest_before=manifest_before, manifest_after=manifest_after,
        semantic_digest_preserved=semantic_digest_preserved, rebuilt_partition_keys=tuple(sorted(touched_partition_keys)),
    )


def compact_feature_dataset(*, repository: MarketDataRepository, dataset_key: DatasetKey, partitioning: PartitioningSpec, compaction_time: datetime) -> CompactionResult:
    if dataset_key.dataset_kind is not DatasetKind.DERIVED_FEATURES:
        raise RepositoryCorruptionError("compact_feature_dataset requires a DERIVED_FEATURES dataset_key")
    require_tz_aware(compaction_time, field_name="compaction_time")
    assert dataset_key.feature_name is not None and dataset_key.feature_version is not None
    manifest_before = repository.manifest_store.read_current(dataset_key)
    if manifest_before is None:
        raise RepositoryCorruptionError(f"cannot compact {dataset_key!r}: no manifest exists yet")
    raw_source_dataset_id = manifest_before.raw_source_dataset_id
    assert raw_source_dataset_id is not None

    records = repository.feature_store.read_records(dataset_key.feature_name, dataset_key.feature_version, dataset_key.instrument_id)
    all_members = [(r.feature_id, r.timestamp) for r in records]
    touched_partition_keys = {partition_key_for(t, partitioning) for _, t in all_members}
    if touched_partition_keys:
        rebuild_touched_partitions(repository=repository, dataset_key=dataset_key, partitioning=partitioning, all_members=all_members, touched_partition_keys=touched_partition_keys)

    manifest_after = rebuild_feature_dataset_manifest(
        repository=repository, dataset_key=dataset_key, partitioning=partitioning, raw_source_dataset_id=raw_source_dataset_id, creation_time=compaction_time,
    )
    semantic_digest_preserved = manifest_before.semantic_digest == manifest_after.semantic_digest
    return CompactionResult(
        dataset_key=dataset_key, manifest_before=manifest_before, manifest_after=manifest_after,
        semantic_digest_preserved=semantic_digest_preserved, rebuilt_partition_keys=tuple(sorted(touched_partition_keys)),
    )
