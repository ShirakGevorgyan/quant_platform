"""Time-safe fold generation for walk-forward EXECUTION (Milestone 4B).

REUSE, NOT REIMPLEMENTATION
--------------------------------------------------------------------------
All purge/embargo/expanding/rolling arithmetic here delegates to
`validation.walk_forward.PurgedWalkForwardSplitter` -- the same,
already-tested engine `features.splitting.build_walk_forward_splits`
wraps for Milestone 3's dataset-BUILD-time splits. This module answers a
DIFFERENT question: Milestone 3's `SplitPlan` is baked into a research
dataset's immutable content at BUILD time (one fixed split, or one fixed
set of folds, chosen once); this module lets an EXECUTION run choose its
OWN walk-forward configuration (n_splits/test_size/purge/embargo/train-
size cap) independently, over the FULL reconstructed timeline of an
already-built dataset, so many different walk-forward configurations can
be evaluated against the same stored data without rebuilding it. Nothing
here duplicates `PurgedWalkForwardSplitter`'s gap arithmetic -- every
generator function is a thin, explicitly-purposed wrapper around it, plus
this module's own (new) concerns: an optional per-fold train/validation
carve-out, grouped and blocked convenience constructors, and bars/
timedelta/calendar-day purge/embargo declarations.

WHY A SEPARATE `Fold`/`FoldPlan`, NOT `features.splitting.DatasetSplit`/
`SplitPlan`
--------------------------------------------------------------------------
`DatasetSplit`/`SplitPlan` are Milestone 3's DURABLE, per-dataset-content
records (named splits, recorded in a manifest). `Fold`/`FoldPlan` here
are EXECUTION-transient: built fresh for one run, over one reconstructed
timeline, never persisted directly (the immutable, persisted-per-fold
record is `execution.results.FoldResult`, built FROM a `Fold` after it
runs). Keeping them distinct avoids overloading Milestone 3's manifest-
oriented types with execution-only fields (an optional validation carve-
out, a group label) that have no meaning in a stored dataset manifest.

A NOTE ON DATACLASS EQUALITY
--------------------------------------------------------------------------
`Fold`/`FoldPlan` hold raw `numpy.ndarray` fields. Comparing two `Fold`
instances with `==` inherits the same landmine `features.splitting.
DatasetSplit` already has (a multi-element array `==` comparison raises
`ValueError: truth value of an array... is ambiguous` inside the
dataclass-generated `__eq__`) -- by design, never worked around here,
consistent with that existing precedent. Callers/tests must compare via
explicit field access (`np.array_equal(...)`), never bare `==`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import FoldValidationError
from quant_platform.core.types import Timeframe
from quant_platform.features.manifests import ResearchDatasetStore
from quant_platform.ml.models import SplitBinding
from quant_platform.validation.walk_forward import PurgedWalkForwardSplitter

_EMPTY_INDICES = np.array([], dtype=np.int64)


def _resolve_bars(
    *, bars: int | None, timedelta: pd.Timedelta | None, calendar_days: int | None,
    timeframe: Timeframe, field_name: str,
) -> int:
    """Converts exactly one of `bars`/`timedelta`/`calendar_days` into an
    equivalent whole-bar count for `timeframe`, always ROUNDING UP (never
    under-purging/under-embargoing a partial bar). `calendar_days` is a
    plain calendar-day count (`pandas.Timedelta(days=N)`) -- NOT a
    trading-session-aware business-day calendar (weekends/holidays are
    not excluded); see this module's own docstring / the milestone's
    documentation for that explicit, deliberate limitation."""
    provided = [v for v in (bars, timedelta, calendar_days) if v is not None]
    if len(provided) != 1:
        raise ValueError(f"{field_name}: exactly one of bars/timedelta/calendar_days must be given, got {provided}")
    if bars is not None:
        if bars < 0:
            raise ValueError(f"{field_name}.bars must be >= 0, got {bars}")
        return bars
    bar_duration = pd.Timedelta(timeframe.duration)
    if timedelta is not None:
        if timedelta < pd.Timedelta(0):
            raise ValueError(f"{field_name}.timedelta must be >= 0, got {timedelta}")
        if timedelta == pd.Timedelta(0):
            return 0
        return int(math.ceil(timedelta / bar_duration))  # noqa: RUF046 -- pandas-stubs types Timedelta.__truediv__ loosely; mypy needs the explicit int()
    assert calendar_days is not None
    if calendar_days < 0:
        raise ValueError(f"{field_name}.calendar_days must be >= 0, got {calendar_days}")
    if calendar_days == 0:
        return 0
    return int(math.ceil(pd.Timedelta(days=calendar_days) / bar_duration))  # noqa: RUF046 -- see above


@dataclass(frozen=True, slots=True)
class PurgeSpec:
    """Configurable purge window -- exactly one of `bars`/`timedelta`/
    `calendar_days` must be given. Purging removes training samples whose
    label window would otherwise overlap the test period (see
    `validation.walk_forward`'s module docstring)."""

    bars: int | None = None
    timedelta: pd.Timedelta | None = None
    calendar_days: int | None = None

    def resolve_bars(self, timeframe: Timeframe) -> int:
        return _resolve_bars(
            bars=self.bars, timedelta=self.timedelta, calendar_days=self.calendar_days,
            timeframe=timeframe, field_name="PurgeSpec",
        )


@dataclass(frozen=True, slots=True)
class EmbargoSpec:
    """Configurable embargo period -- exactly one of `bars`/`timedelta`/
    `calendar_days` must be given. Embargo adds a further buffer after
    the purge window, guarding against serial correlation beyond the
    label horizon itself."""

    bars: int | None = None
    timedelta: pd.Timedelta | None = None
    calendar_days: int | None = None

    def resolve_bars(self, timeframe: Timeframe) -> int:
        return _resolve_bars(
            bars=self.bars, timedelta=self.timedelta, calendar_days=self.calendar_days,
            timeframe=timeframe, field_name="EmbargoSpec",
        )


@dataclass(frozen=True, slots=True)
class Fold:
    """One execution-transient fold: train (optionally further carved
    into train + validation) and test row POSITIONS into a reconstructed
    timeline DataFrame (`iloc`-ready, not label-based). See module
    docstring for why this is a distinct type from `features.splitting.
    DatasetSplit` and why `==` comparison is unsafe (numpy array fields)."""

    fold_index: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    validation_indices: np.ndarray = field(default_factory=lambda: _EMPTY_INDICES)
    validation_start: pd.Timestamp | None = None
    validation_end: pd.Timestamp | None = None
    group: str | None = None

    def __post_init__(self) -> None:
        if self.fold_index < 0:
            raise FoldValidationError(f"Fold.fold_index must be >= 0, got {self.fold_index}")
        if len(self.train_indices) == 0:
            raise FoldValidationError(f"Fold {self.fold_index}: train_indices must not be empty")
        if len(self.test_indices) == 0:
            raise FoldValidationError(f"Fold {self.fold_index}: test_indices must not be empty")


def required_label_purge_bars_for(label_horizon_bars: int) -> int:
    """The MINIMUM number of bars that must separate the last training
    row from the first validation/test row, given a forward-looking
    label computed `label_horizon_bars` bars ahead of each row (see
    `features.labels`' module docstring: every label kind -- future
    return, log return, binary direction, vol-adjusted return, triple
    barrier -- resolves using price data through row `i + horizon_bars`
    inclusive, never beyond it; `build_triple_barrier_labels` loops
    `offset in range(1, horizon_bars + 1)`, and `compute_future_return`/
    `compute_future_log_return`/`compute_binary_direction`/
    `compute_vol_adjusted_return` all call `close.shift(-horizon_bars)` --
    every kind's forward window is EXACTLY `[i+1, i+horizon_bars]`, never
    `i+horizon_bars+1` or beyond).

    EXACT OFF-BY-ONE PROOF
    --------------------------------------------------------------------------
    Let `T` be the last kept training row position and `S` the first
    validation/test row position (`S > T`). Row `T`'s label uses price
    data through row `T + horizon_bars`. No leak requires that row to
    still be BEFORE `S`:

        T + horizon_bars < S
        S - T > horizon_bars
        (S - T - 1) >= horizon_bars                    [integers]

    This codebase defines "gap" as the count of SKIPPED rows strictly
    between the last train row and the first validation/test row --
    exactly `S - T - 1` (see `execution_validation._validate_purge_embargo`
    and `validation.walk_forward.PurgedWalkForwardSplitter`, whose own
    `train_end = test_start - gap` convention makes the skipped span
    exactly `gap` rows). Substituting: `gap >= horizon_bars`. So the
    required minimum is **exactly `horizon_bars`, not `horizon_bars - 1`**
    -- a gap of `horizon_bars - 1` leaves row `T`'s label depending on row
    `T + horizon_bars = S`, the first validation/test row itself: a
    genuine leak. `tests/unit/execution/test_label_horizon_purge.py`
    proves both edges directly against the real label computation
    functions, not just this arithmetic in isolation.

    This is an IDENTITY function today (the minimum purge always equals
    the horizon exactly, for every label kind this codebase implements)
    -- kept as its own named, documented function rather than inlined
    `= label_horizon_bars` so a hypothetical FUTURE label kind with
    different forward-window semantics has one obvious place to change
    the formula, and so every caller states its INTENT ("the purge
    required BY the label"), not just a bare number.
    """
    if label_horizon_bars < 0:
        raise ValueError(f"label_horizon_bars must be >= 0, got {label_horizon_bars}")
    return label_horizon_bars


@dataclass(frozen=True, slots=True)
class FoldPlan:
    """An ordered sequence of `Fold`s generated for one execution run,
    plus the configuration that produced them (recorded for the
    execution manifest/report, never re-derived by guesswork).

    `purge_bars` is the DECLARED gap actually used to carve every fold's
    boundaries (from `SplitBinding.params`, identity-relevant, unchanged
    by this milestone's leakage fix). `label_horizon_bars` is the TRUE,
    dataset-manifest-derived fact; `required_label_purge_bars` is the
    POLICY value derived from it (`required_label_purge_bars_for`, see
    that function's docstring for the exact off-by-one proof). This
    milestone's policy is REJECTION, not silent widening (see
    `execution_validation._validate_label_horizon_purge` and
    `docs/execution_engine.md`'s "Label-information purge" section) --
    so whenever a `FoldPlan` reaches `COMPLETED`, `purge_bars >=
    required_label_purge_bars` is already guaranteed; `purge_bars` IS the
    effective purge in every accepted execution, recorded a second,
    explicitly-named time on `ExecutionManifest` for audit purposes."""

    strategy: str
    folds: tuple[Fold, ...]
    purge_bars: int
    embargo_bars: int
    total_rows: int
    label_horizon_bars: int
    required_label_purge_bars: int

    def __post_init__(self) -> None:
        if not self.folds:
            raise FoldValidationError("FoldPlan must contain at least one fold")
        indices = [f.fold_index for f in self.folds]
        if indices != list(range(len(indices))):
            raise FoldValidationError(
                f"FoldPlan.folds fold_index values must be exactly 0..N-1 in order, got {indices}"
            )
        if self.purge_bars < 0 or self.embargo_bars < 0:
            raise FoldValidationError("FoldPlan.purge_bars/embargo_bars must be >= 0")
        if self.label_horizon_bars < 0:
            raise FoldValidationError("FoldPlan.label_horizon_bars must be >= 0")
        if self.required_label_purge_bars != required_label_purge_bars_for(self.label_horizon_bars):
            raise FoldValidationError(  # pragma: no cover - only reachable via a hand-built, inconsistent FoldPlan
                "FoldPlan.required_label_purge_bars is inconsistent with label_horizon_bars"
            )


def _carve_validation(
    train_indices: np.ndarray, *, purge_bars: int, validation_fraction: float, index: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp | None, pd.Timestamp | None]:
    """Best-effort: carve the CHRONOLOGICAL TAIL of `train_indices` off as
    a validation slice, separated from the (now-shorter) train slice by
    the SAME `purge_bars` gap already used at the train/test boundary --
    never new purge math, just the identical gap applied a second time.
    Returns `(train_indices, validation_indices, validation_start,
    validation_end)`; if there is not enough room, returns the ORIGINAL
    train_indices unchanged and an empty validation slice -- never raises,
    since a validation carve-out is a best-effort convenience, not a
    correctness requirement."""
    if validation_fraction <= 0.0:
        return train_indices, _EMPTY_INDICES, None, None
    n = len(train_indices)
    raw_validation_size = int(n * validation_fraction)
    if raw_validation_size <= 0:
        return train_indices, _EMPTY_INDICES, None, None
    validation_start_pos = n - raw_validation_size
    train_end_pos = validation_start_pos - purge_bars
    if train_end_pos <= 0:
        return train_indices, _EMPTY_INDICES, None, None
    new_train = train_indices[:train_end_pos]
    validation_idx = train_indices[validation_start_pos:]
    if len(validation_idx) == 0:
        return train_indices, _EMPTY_INDICES, None, None
    return new_train, validation_idx, index[validation_idx[0]], index[validation_idx[-1]]


def _build_fold(
    fold_index: int, wf_split_train: np.ndarray, wf_split_test: np.ndarray, *, index: pd.DatetimeIndex,
    purge_bars: int, validation_fraction: float, group: str | None = None,
) -> Fold:
    train_indices, validation_indices, validation_start, validation_end = _carve_validation(
        wf_split_train, purge_bars=purge_bars, validation_fraction=validation_fraction, index=index,
    )
    return Fold(
        fold_index=fold_index,
        train_indices=train_indices, test_indices=wf_split_test,
        train_start=index[train_indices[0]], train_end=index[train_indices[-1]],
        test_start=index[wf_split_test[0]], test_end=index[wf_split_test[-1]],
        validation_indices=validation_indices, validation_start=validation_start, validation_end=validation_end,
        group=group,
    )


def _generate_walk_forward_folds(
    timestamps: pd.Series, *, n_splits: int, test_size: int, purge_bars: int, embargo_bars: int,
    max_train_size: int | None, validation_fraction: float, strategy: str, label_horizon_bars: int,
) -> FoldPlan:
    splitter = PurgedWalkForwardSplitter(
        n_splits=n_splits, test_size=test_size, label_horizon=purge_bars, embargo=embargo_bars,
        max_train_size=max_train_size,
    )
    index = pd.DatetimeIndex(pd.Index(timestamps))
    folds = tuple(
        _build_fold(
            wf.fold, wf.train_indices, wf.test_indices, index=index,
            purge_bars=purge_bars, validation_fraction=validation_fraction,
        )
        for wf in splitter.split(timestamps)
    )
    return FoldPlan(
        strategy=strategy, folds=folds, purge_bars=purge_bars, embargo_bars=embargo_bars, total_rows=len(index),
        label_horizon_bars=label_horizon_bars, required_label_purge_bars=required_label_purge_bars_for(label_horizon_bars),
    )


def generate_expanding_folds(
    timestamps: pd.Series, *, n_splits: int, test_size: int, label_horizon_bars: int, purge_bars: int = 0,
    embargo_bars: int = 0, validation_fraction: float = 0.0,
) -> FoldPlan:
    """Expanding window: every fold's training data starts at row 0 and
    grows -- never discards early history. `label_horizon_bars` is
    REQUIRED (no default) -- it must come from the research dataset
    manifest's own `label_definition`, never guessed or defaulted to 0,
    since 0 would silently disable the label-information purge check
    (see `execution_validation._validate_label_horizon_purge`)."""
    return _generate_walk_forward_folds(
        timestamps, n_splits=n_splits, test_size=test_size, purge_bars=purge_bars, embargo_bars=embargo_bars,
        max_train_size=None, validation_fraction=validation_fraction, strategy="expanding_walk_forward",
        label_horizon_bars=label_horizon_bars,
    )


def generate_rolling_folds(
    timestamps: pd.Series, *, n_splits: int, test_size: int, max_train_size: int, label_horizon_bars: int,
    purge_bars: int = 0, embargo_bars: int = 0, validation_fraction: float = 0.0,
) -> FoldPlan:
    """Rolling (fixed-size) window: every fold's training data is capped
    at `max_train_size` rows, sliding forward -- old history is dropped
    as new folds advance. `label_horizon_bars` is REQUIRED -- see
    `generate_expanding_folds`."""
    return _generate_walk_forward_folds(
        timestamps, n_splits=n_splits, test_size=test_size, purge_bars=purge_bars, embargo_bars=embargo_bars,
        max_train_size=max_train_size, validation_fraction=validation_fraction, strategy="rolling_walk_forward",
        label_horizon_bars=label_horizon_bars,
    )


def generate_blocked_time_folds(
    timestamps: pd.Series, *, n_blocks: int, label_horizon_bars: int, purge_bars: int = 0, embargo_bars: int = 0,
    validation_fraction: float = 0.0,
) -> FoldPlan:
    """Blocked time split: the series is divided into `n_blocks`
    contiguous, roughly-equal blocks; block 0 is reserved as pure history
    and each subsequent block is, in turn, one fold's test block, with
    every earlier block as training data. Implemented as a direct,
    explicit configuration of the SAME expanding walk-forward engine
    (`test_size = len(timestamps) // n_blocks`, `n_splits = n_blocks -
    1`) -- a convenience entry point, not a new splitting algorithm.
    `label_horizon_bars` is REQUIRED -- see `generate_expanding_folds`."""
    if n_blocks < 2:
        raise ValueError(f"n_blocks must be >= 2 (at least one history block and one test block), got {n_blocks}")
    n = len(timestamps)
    test_size = n // n_blocks
    if test_size <= 0:
        raise ValueError(f"n_blocks={n_blocks} is too large for {n} rows (test_size would be 0)")
    plan = generate_expanding_folds(
        timestamps, n_splits=n_blocks - 1, test_size=test_size, purge_bars=purge_bars, embargo_bars=embargo_bars,
        validation_fraction=validation_fraction, label_horizon_bars=label_horizon_bars,
    )
    return FoldPlan(
        strategy="blocked_time_split", folds=plan.folds, purge_bars=purge_bars, embargo_bars=embargo_bars,
        total_rows=plan.total_rows, label_horizon_bars=plan.label_horizon_bars,
        required_label_purge_bars=plan.required_label_purge_bars,
    )


def generate_grouped_walk_forward_folds(
    timeline: pd.DataFrame, *, timestamp_column: str, group_column: str, n_splits: int, test_size: int,
    label_horizon_bars: int, purge_bars: int = 0, embargo_bars: int = 0, validation_fraction: float = 0.0,
) -> FoldPlan:
    """Grouped walk forward: an independent expanding walk-forward split
    is generated WITHIN each distinct value of `group_column` (e.g. a
    per-instrument key in a multi-symbol dataset), each group's fold
    indices mapped back to row POSITIONS in the full `timeline` (never
    assuming a group's rows are contiguous within it). Fold indices are
    renumbered sequentially across all groups, ordered by
    `(group value, that group's own fold number)`, so `FoldPlan.folds`
    still satisfies the plain `0..N-1` ordering invariant every other
    plan does. `label_horizon_bars` is REQUIRED -- see
    `generate_expanding_folds`."""
    if group_column not in timeline.columns:
        raise ValueError(f"group_column {group_column!r} is not a column of timeline")
    if timestamp_column not in timeline.columns:
        raise ValueError(f"timestamp_column {timestamp_column!r} is not a column of timeline")

    index = pd.DatetimeIndex(pd.Index(timeline[timestamp_column]))
    all_folds: list[Fold] = []
    next_index = 0
    for group_value, group_df in timeline.groupby(group_column, sort=True):
        local_positions = timeline.index.get_indexer(group_df.index)
        group_timestamps = group_df[timestamp_column]
        splitter = PurgedWalkForwardSplitter(
            n_splits=n_splits, test_size=test_size, label_horizon=purge_bars, embargo=embargo_bars,
        )
        for wf in splitter.split(group_timestamps):
            global_train = local_positions[wf.train_indices]
            global_test = local_positions[wf.test_indices]
            all_folds.append(
                _build_fold(
                    next_index, global_train, global_test, index=index,
                    purge_bars=purge_bars, validation_fraction=validation_fraction, group=str(group_value),
                )
            )
            next_index += 1

    if not all_folds:
        raise FoldValidationError(f"No folds could be generated for any value of group_column={group_column!r}")
    return FoldPlan(
        strategy="grouped_walk_forward", folds=tuple(all_folds), purge_bars=purge_bars, embargo_bars=embargo_bars,
        total_rows=len(timeline), label_horizon_bars=label_horizon_bars,
        required_label_purge_bars=required_label_purge_bars_for(label_horizon_bars),
    )


_JsonPrimitive = str | int | float | bool | None


def _param_int(params: dict[str, _JsonPrimitive], key: str, *, default: int | None = None) -> int:
    value = params.get(key, default)
    if value is None:
        raise ValueError(f"split_binding.params[{key!r}] is required for this strategy")
    return int(value)


def _param_float(params: dict[str, _JsonPrimitive], key: str, *, default: float) -> float:
    value = params.get(key, default)
    if value is None:
        raise ValueError(f"split_binding.params[{key!r}] must not be null")
    return float(value)


def build_folds_from_split_binding(
    split_binding: SplitBinding, timestamps: pd.Series, *, label_horizon_bars: int,
) -> FoldPlan:
    """Dispatches on `split_binding.strategy` (reusing `ExperimentSpec`'s
    OWN existing, identity-relevant field -- Milestone 4B introduces no
    new identity-relevant configuration) to the matching generator above.
    `split_binding.params` supplies each strategy's numeric arguments as
    JSON primitives (see `ml.models.SplitBinding`).

    `label_horizon_bars` is a REQUIRED keyword-only argument and is
    deliberately NOT read from `split_binding.params`: it is a fact about
    the bound research dataset's `LabelDefinition` (see
    `runner._required_label_horizon_bars`), not a user-declared split
    parameter. `SplitBinding` gains no new field for it -- callers must
    derive it from the already-loaded, immutable `ResearchDatasetManifest`
    and pass it in explicitly. Passing a value that does not actually
    match the bound dataset's real label horizon would defeat the
    label-information purge check in `execution_validation`, so this
    function trusts its caller to have sourced it correctly; it does not
    (and cannot, from `split_binding` alone) re-derive or verify it."""
    params = dict(split_binding.params)
    purge_bars = _param_int(params, "purge_bars", default=0)
    embargo_bars = _param_int(params, "embargo_bars", default=0)
    validation_fraction = _param_float(params, "validation_fraction", default=0.0)

    strategy = split_binding.strategy
    if strategy == "expanding_walk_forward":
        return generate_expanding_folds(
            timestamps, n_splits=_param_int(params, "n_splits"), test_size=_param_int(params, "test_size"),
            purge_bars=purge_bars, embargo_bars=embargo_bars, validation_fraction=validation_fraction,
            label_horizon_bars=label_horizon_bars,
        )
    if strategy == "rolling_walk_forward":
        return generate_rolling_folds(
            timestamps, n_splits=_param_int(params, "n_splits"), test_size=_param_int(params, "test_size"),
            max_train_size=_param_int(params, "max_train_size"), purge_bars=purge_bars, embargo_bars=embargo_bars,
            validation_fraction=validation_fraction, label_horizon_bars=label_horizon_bars,
        )
    if strategy == "blocked_time_split":
        return generate_blocked_time_folds(
            timestamps, n_blocks=_param_int(params, "n_blocks"), purge_bars=purge_bars, embargo_bars=embargo_bars,
            validation_fraction=validation_fraction, label_horizon_bars=label_horizon_bars,
        )
    raise ValueError(
        f"Unsupported split_binding.strategy for execution: {strategy!r} -- expected one of "
        "'expanding_walk_forward', 'rolling_walk_forward', 'blocked_time_split' (grouped walk-forward "
        "requires a group column and is invoked directly via generate_grouped_walk_forward_folds, not "
        "through this dispatcher, since ExperimentSpec has no generic place to declare one yet)"
    )


def reconstruct_dataset_timeline(
    research_store: ResearchDatasetStore, *, dataset_id: str, content_id: str, timestamp_column: str = "open_time",
) -> pd.DataFrame:
    """Reassembles a research dataset's FULL, chronologically-ordered
    timeline from every split its manifest recorded (train/validation/
    test, or fold_k_train/fold_k_test/..., whatever the dataset was built
    with) -- concatenated and sorted by `timestamp_column`. This is safe
    specifically because every split `features.dataset_builder.
    ResearchDatasetBuilder` produces is non-overlapping and internally
    sorted; the duplicate/monotonicity checks below are a defense-in-
    depth verification of that invariant, not a workaround for a known
    violation of it. A dataset built by some OTHER, non-conforming path
    with genuinely overlapping splits would fail loudly here
    (`FoldValidationError`) rather than silently double-counting rows."""
    splits = research_store.read_artifacts(dataset_id, content_id)
    if splits is None:
        raise FoldValidationError(
            f"No stored content for dataset_id={dataset_id!r} content_id={content_id!r}",
            context={"dataset_id": dataset_id, "content_id": content_id},
        )
    if not splits:
        raise FoldValidationError(
            f"Dataset dataset_id={dataset_id!r} content_id={content_id!r} has no named splits to reconstruct",
            context={"dataset_id": dataset_id, "content_id": content_id},
        )
    combined = pd.concat(list(splits.values()), axis=0, ignore_index=True)
    if timestamp_column not in combined.columns:
        raise FoldValidationError(f"Reconstructed timeline is missing expected column {timestamp_column!r}")
    if combined[timestamp_column].isna().any():
        raise FoldValidationError(f"Reconstructed timeline contains null {timestamp_column!r} value(s)")
    combined = combined.sort_values(timestamp_column, kind="mergesort").reset_index(drop=True)

    ts = combined[timestamp_column]
    if ts.duplicated().any():
        dup_count = int(ts.duplicated().sum())
        raise FoldValidationError(
            f"Reconstructed dataset timeline has {dup_count} duplicate {timestamp_column!r} value(s) -- "
            "the underlying dataset splits are not disjoint; refusing to build folds over ambiguous rows",
            context={"duplicate_count": dup_count, "dataset_id": dataset_id, "content_id": content_id},
        )
    if len(ts) > 1 and not pd.DatetimeIndex(pd.Index(ts)).is_monotonic_increasing:
        raise FoldValidationError(  # pragma: no cover - unreachable after an explicit sort with no NaT/dups
            "Reconstructed dataset timeline is not monotonically increasing after sorting"
        )
    return combined


def fold_row_counts(fold: Fold) -> tuple[int, int, int]:
    """`(train_size, validation_size, test_size)` for one fold."""
    return len(fold.train_indices), len(fold.validation_indices), len(fold.test_indices)


def iter_fold_bounds(plan: FoldPlan) -> Sequence[tuple[int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """`(fold_index, train_start, train_end, test_start, test_end)` for
    every fold in `plan`, in order -- a convenience for `timeline.py`."""
    return [(f.fold_index, f.train_start, f.train_end, f.test_start, f.test_end) for f in plan.folds]


__all__ = [
    "EmbargoSpec",
    "Fold",
    "FoldPlan",
    "PurgeSpec",
    "build_folds_from_split_binding",
    "fold_row_counts",
    "generate_blocked_time_folds",
    "generate_expanding_folds",
    "generate_grouped_walk_forward_folds",
    "generate_rolling_folds",
    "iter_fold_bounds",
    "reconstruct_dataset_timeline",
    "required_label_purge_bars_for",
]
