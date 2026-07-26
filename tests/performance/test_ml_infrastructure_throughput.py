"""Performance benchmarks for the Milestone 4A ML core infrastructure
(Section 21). Same philosophy as `test_feature_throughput.py`/
`test_historical_pipeline_throughput.py`: conservative floors (roughly
10x-100x below measured numbers on reference hardware) to catch a severe
accidental regression (e.g. an O(n^2) identity computation, a per-call
disk fsync where none is needed) without being flaky on a slower CI
runner -- these are NOT production throughput guarantees, and integrity
verification (content-hash recomputation on every artifact read) is
never skipped to make a number look better.

Measured on reference hardware (informational; one real run of this
file's own benchmarks, Windows 11 / NTFS; expect run-to-run variance of
at least +/-30%, and note results may be affected by OS filesystem
caching -- a warm page/inode cache after the first iteration of a loop
is not disclaimed away, since it reflects a realistic repeated-run
scenario). Filesystem-touching operations (artifact writes, event
appends) are visibly slower than pure in-memory ones on this platform,
dominated by per-call file-lock/rename overhead rather than the actual
bytes moved -- expected, and exactly what these floors guard against
regressing further:
  - `ExperimentSpec.to_identity_payload()` canonical JSON serialization
    (2 features, typical binding sizes), 2,000 iterations: 0.009ms/iter
    median, ~112,000 iter/sec.
  - `compute_experiment_identity` (includes the above serialization plus
    SHA-256 hashing), 2,000 iterations: 0.012ms/iter median, ~80,600
    iter/sec.
  - `MLArtifactStore.write_artifact` (10 KB payload, unique content per
    call so every write is a real, uncached write -- content file +
    metadata sidecar, each its own temp-file-then-rename), 500
    iterations: 2.17ms/iter median, ~461 iter/sec.
  - `MLArtifactStore.read_artifact` (10 KB payload, includes SHA-256
    re-verification every read -- never skipped), 500 iterations:
    0.38ms/iter median, ~2,600 iter/sec.
  - `ExperimentPreparer.prepare` (full preparation pipeline: resolve
    model+dataset, capture environment, create manifest, validate, write
    validation-report artifact, transition to ready), 50 DISTINCT
    experiments (never idempotent no-ops, which would be near-instant):
    16.7ms/iter median, ~60 iter/sec.
  - `ExperimentManifestStore.load` (reconstruction from disk), 200
    iterations against one already-created manifest: 0.33ms/iter
    median, ~3,070 iter/sec.
  - `ExperimentEventStore.append`, 500 sequential appends to one growing
    log: 3.68ms/iter median, ~272 iter/sec (dominated by per-call
    `DatasetLock` file-lock acquire/release, plus re-reading and
    re-parsing the entire growing log every call by design -- see
    `tracking.py`'s module docstring; this is why per-append latency
    grows mildly with log length rather than staying flat).
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest
from tests.unit.ml.conftest import build_registry, make_dataset_manifest, make_experiment_spec_kwargs

from quant_platform.features.manifests import ResearchManifestStore
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.experiment_identity import compute_experiment_identity
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.models import ArtifactCategory, CodeRevisionBinding, ModelHyperparameters
from quant_platform.ml.persistence import canonical_json_bytes
from quant_platform.ml.tracking import EventType, ExperimentEventStore

pytestmark = pytest.mark.performance


def _timed_iterations(fn, count: int) -> list[float]:
    timings = []
    for _ in range(count):
        started = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - started)
    return timings


def _report(label: str, timings: list[float]) -> float:
    median = statistics.median(timings)
    rate = 1.0 / median if median > 0 else float("inf")
    print(f"\n{label}: n={len(timings)} median={median * 1000:.3f}ms p95={sorted(timings)[int(len(timings) * 0.95)] * 1000:.3f}ms rate={rate:,.0f}/sec")
    return median


class TestSpecSerializationThroughput:
    def test_identity_payload_serialization(self) -> None:
        spec = ExperimentSpec(**make_experiment_spec_kwargs())

        median = _report("ExperimentSpec.to_identity_payload + canonical_json_bytes", _timed_iterations(
            lambda: canonical_json_bytes(spec.to_identity_payload()), 2000
        ))
        assert median < 0.01, "canonical serialization of one spec should not take >10ms (100x the measured floor)"


class TestIdentityComputationThroughput:
    def test_compute_experiment_identity(self) -> None:
        spec = ExperimentSpec(**make_experiment_spec_kwargs())

        median = _report("compute_experiment_identity", _timed_iterations(lambda: compute_experiment_identity(spec), 2000))
        assert median < 0.01


class TestArtifactStoreThroughput:
    def test_write_artifact(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        payload = b"x" * 10_000
        counter = {"i": 0}

        def write() -> None:
            counter["i"] += 1
            # Unique content per call so every write is a real, uncached write (never dedup-short-circuited).
            store.write_artifact(payload + counter["i"].to_bytes(8, "big"), category=ArtifactCategory.MODEL)

        median = _report("MLArtifactStore.write_artifact (10KB, unique content)", _timed_iterations(write, 500))
        assert median < 0.05, "a single 10KB artifact write should not take >50ms"

    def test_read_artifact(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        ref = store.write_artifact(b"y" * 10_000, category=ArtifactCategory.MODEL)

        median = _report("MLArtifactStore.read_artifact (10KB, hash reverified every read)", _timed_iterations(
            lambda: store.read_artifact(ref.content_hash), 500
        ))
        assert median < 0.02, "a single 10KB artifact read+verify should not take >20ms"


class TestExperimentPreparationThroughput:
    def test_prepare_distinct_experiments(self, tmp_path: Path) -> None:
        research_store = ResearchManifestStore(tmp_path / "research")
        research_store.save(make_dataset_manifest())
        preparer = ExperimentPreparer(
            ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=build_registry(),
            research_manifest_store=research_store,
        )
        counter = {"i": 0}

        def prepare() -> None:
            counter["i"] += 1
            spec = ExperimentSpec(**make_experiment_spec_kwargs(
                hyperparameters=ModelHyperparameters(values={"alpha": float(counter["i"])}),
                code_revision_binding=CodeRevisionBinding(revision=f"{counter['i']:040x}", source="git", is_dirty=True),
            ))
            preparer.prepare(spec)

        median = _report("ExperimentPreparer.prepare (distinct experiments, full pipeline)", _timed_iterations(prepare, 50))
        assert median < 0.5, "a full experiment preparation should not take >500ms"


class TestManifestReconstructionThroughput:
    def test_manifest_load(self, tmp_path: Path) -> None:
        research_store = ResearchManifestStore(tmp_path / "research")
        research_store.save(make_dataset_manifest())
        preparer = ExperimentPreparer(
            ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=build_registry(),
            research_manifest_store=research_store,
        )
        manifest = preparer.prepare(ExperimentSpec(**make_experiment_spec_kwargs()))
        experiment_id = manifest.identity.experiment_id

        median = _report("ExperimentManifestStore.load (reconstruction)", _timed_iterations(
            lambda: preparer.manifest_store.load(experiment_id), 200
        ))
        assert median < 0.02, "reconstructing one manifest from disk should not take >20ms"


class TestEventAppendThroughput:
    def test_sequential_appends(self, tmp_path: Path) -> None:
        store = ExperimentEventStore(tmp_path)
        eid = "a" * 64

        median = _report("ExperimentEventStore.append (sequential, growing log)", _timed_iterations(
            lambda: store.append(eid, EventType.ARTIFACT_WRITTEN), 500
        ))
        assert median < 0.05, "a single event append should not take >50ms even as the log grows"
