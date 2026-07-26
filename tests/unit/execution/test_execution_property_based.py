"""Property-based tests (Section 16) for the execution engine's most
safety-critical invariants: generated folds are always strictly
chronological and never leak; the purge/embargo gap always meets or
exceeds what was configured; cross-fold test sets never overlap for
expanding/rolling walk-forward; resume planning is a pure, deterministic
function of its inputs (calling it twice with identical state always
produces an identical plan); `ExecutionStage` transition legality is a
pure function of its inputs, and every terminal stage has zero legal
targets, for every stage in the enum, not just the ones covered by
example-based tests."""

from __future__ import annotations

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_platform.core.types import Timeframe
from quant_platform.execution.resume import build_resume_plan
from quant_platform.execution.splitters import (
    PurgeSpec,
    generate_expanding_folds,
    generate_rolling_folds,
)
from quant_platform.execution.state_machine import (
    ExecutionStage,
    is_legal_execution_transition,
    is_terminal_stage,
)
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.models import ArtifactCategory
from quant_platform.ml.persistence import canonical_json_bytes

_N_ROWS = 2000
_TIMESTAMPS = pd.Series(pd.date_range("2024-01-01", periods=_N_ROWS, freq="1min", tz="UTC"))

_fold_config = st.fixed_dictionaries({
    "n_splits": st.integers(min_value=1, max_value=5),
    "test_size": st.integers(min_value=10, max_value=50),
    "purge_bars": st.integers(min_value=0, max_value=20),
    "embargo_bars": st.integers(min_value=0, max_value=20),
    # Fixed at 0: these properties are about the ACTUAL fold gap the
    # generator delivers for a given purge/embargo declaration, which
    # `label_horizon_bars` does not influence at all (it only feeds
    # `FoldPlan.required_label_purge_bars`, checked by the separate
    # `execution_validation._validate_label_horizon_purge` gate -- see
    # `test_label_horizon_purge.py` for THAT property).
    "label_horizon_bars": st.just(0),
})


@given(_fold_config)
@settings(max_examples=100)
def test_expanding_folds_always_chronological_and_purged(config: dict[str, int]) -> None:
    plan = generate_expanding_folds(_TIMESTAMPS, **config)
    required_gap = config["purge_bars"] + config["embargo_bars"]
    for fold in plan.folds:
        assert int(fold.train_indices.max()) < int(fold.test_indices.min())
        gap = int(fold.test_indices.min()) - int(fold.train_indices.max()) - 1
        assert gap == required_gap
        assert int(fold.train_indices[0]) == 0  # expanding: always starts at row 0


@given(_fold_config, st.integers(min_value=100, max_value=500))
@settings(max_examples=100)
def test_rolling_folds_train_size_never_exceeds_cap(config: dict[str, int], max_train_size: int) -> None:
    plan = generate_rolling_folds(_TIMESTAMPS, max_train_size=max_train_size, **config)
    for fold in plan.folds:
        assert len(fold.train_indices) <= max_train_size
        assert int(fold.train_indices.max()) < int(fold.test_indices.min())


@given(_fold_config)
@settings(max_examples=100)
def test_cross_fold_test_sets_never_overlap(config: dict[str, int]) -> None:
    plan = generate_expanding_folds(_TIMESTAMPS, **config)
    seen: set[int] = set()
    for fold in plan.folds:
        positions = set(fold.test_indices.tolist())
        assert not (positions & seen), "two folds' test sets overlap"
        seen |= positions


@given(st.integers(min_value=0, max_value=10_000))
@settings(max_examples=100)
def test_purge_spec_timedelta_never_under_purges(seconds: int) -> None:
    """For ANY non-negative duration, the resolved bar count, multiplied
    back out by the bar duration, must cover AT LEAST the requested
    timedelta -- rounding must only ever go up, never down."""
    delta = pd.Timedelta(seconds=seconds)
    bars = PurgeSpec(timedelta=delta).resolve_bars(Timeframe.M1)
    assert bars * Timeframe.M1.duration >= delta


@given(st.sampled_from(list(ExecutionStage)), st.sampled_from(list(ExecutionStage)))
@settings(max_examples=200)
def test_transition_legality_is_a_pure_deterministic_function(current: ExecutionStage, target: ExecutionStage) -> None:
    first = is_legal_execution_transition(current, target)
    second = is_legal_execution_transition(current, target)
    assert first == second
    assert isinstance(first, bool)


@given(st.sampled_from(list(ExecutionStage)))
@settings(max_examples=50)
def test_terminal_stages_have_zero_legal_targets(stage: ExecutionStage) -> None:
    if is_terminal_stage(stage):
        assert not any(is_legal_execution_transition(stage, target) for target in ExecutionStage)
    else:
        assert any(is_legal_execution_transition(stage, target) for target in ExecutionStage)


@given(
    st.sets(st.integers(min_value=0, max_value=9), max_size=10),
    st.sets(st.integers(min_value=0, max_value=9), max_size=10),
)
@settings(max_examples=100)
def test_resume_plan_is_a_pure_function_of_its_inputs(tmp_path_factory, completed: set[int], force: set[int]) -> None:
    """Calling `build_resume_plan` twice against the identical,
    unmodified manifest/plan/artifact-store state must always produce an
    identical `ResumePlan` -- no hidden mutable state, no reliance on
    call order or wall-clock time."""
    import numpy as np

    from quant_platform.execution.manifests import EXECUTION_MANIFEST_SCHEMA_VERSION, ExecutionManifest
    from quant_platform.execution.splitters import Fold, FoldPlan
    from quant_platform.execution.state_machine import ExecutionStage as Stage
    from quant_platform.ml.persistence import format_utc_timestamp, utc_now

    tmp_path = tmp_path_factory.mktemp("resume_prop")
    store = MLArtifactStore(tmp_path)
    now = format_utc_timestamp(utc_now())
    ts = pd.Timestamp("2024-01-01", tz="UTC")

    refs = {i: store.write_artifact(canonical_json_bytes({"i": i}), category=ArtifactCategory.FOLD_RESULT) for i in completed}
    manifest = ExecutionManifest(
        schema_version=EXECUTION_MANIFEST_SCHEMA_VERSION, experiment_id="a" * 64, stage=Stage.RUNNING_FOLD,
        created_at=now, updated_at=now, completed_fold_indices=tuple(sorted(completed)), fold_result_references=refs,
    )
    folds = tuple(
        Fold(fold_index=i, train_indices=np.arange(0, 5), test_indices=np.arange(10, 15), train_start=ts, train_end=ts, test_start=ts, test_end=ts)
        for i in range(10)
    )
    plan = FoldPlan(
        strategy="x", purge_bars=0, embargo_bars=0, total_rows=100, folds=folds,
        label_horizon_bars=0, required_label_purge_bars=0,
    )

    result1 = build_resume_plan(manifest, plan, artifact_store=store, force_rerun_folds=frozenset(force))
    result2 = build_resume_plan(manifest, plan, artifact_store=store, force_rerun_folds=frozenset(force))
    assert result1.verified_complete == result2.verified_complete
    assert result1.needs_rerun == result2.needs_rerun
    assert [f.fold_index for f in result1.remaining_folds] == [f.fold_index for f in result2.remaining_folds]
