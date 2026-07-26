"""Time-aware dataset splitting -- chronological train/validation/test, and
expanding/rolling purged walk-forward (wrapping the existing, already-tested
`validation.walk_forward.PurgedWalkForwardSplitter` rather than
reimplementing purge/embargo arithmetic a second time). Random shuffling is
never offered as an option anywhere in this module -- there is no code path
that reorders rows before splitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import ValidationSplitError
from quant_platform.validation.walk_forward import PurgedWalkForwardSplitter


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    name: str
    indices: np.ndarray
    start: pd.Timestamp
    end: pd.Timestamp

    def to_json_dict(self) -> dict[str, object]:
        return {"name": self.name, "row_count": len(self.indices), "start": self.start.isoformat(), "end": self.end.isoformat()}


@dataclass(frozen=True, slots=True)
class SplitPlan:
    strategy: str
    splits: tuple[DatasetSplit, ...]
    purge_bars: int
    embargo_bars: int
    gap_bars: int

    def get(self, name: str) -> DatasetSplit:
        for split in self.splits:
            if split.name == name:
                return split
        raise KeyError(f"No split named {name!r} in this plan (have: {[s.name for s in self.splits]})")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "purge_bars": self.purge_bars,
            "embargo_bars": self.embargo_bars,
            "gap_bars": self.gap_bars,
            "splits": [s.to_json_dict() for s in self.splits],
        }


def _validate_fractions(train_fraction: float, validation_fraction: float) -> None:
    if not (0.0 < train_fraction < 1.0):
        raise ValidationSplitError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not (0.0 < validation_fraction < 1.0):
        raise ValidationSplitError(f"validation_fraction must be in (0, 1), got {validation_fraction}")
    if train_fraction + validation_fraction >= 1.0:
        raise ValidationSplitError(
            f"train_fraction + validation_fraction must be < 1 (leaving room for a test split), "
            f"got {train_fraction} + {validation_fraction} = {train_fraction + validation_fraction}"
        )


def build_chronological_split(
    timestamps: pd.Series,
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    purge_bars: int = 0,
    embargo_bars: int = 0,
    gap_bars: int = 0,
) -> SplitPlan:
    """A single chronological train -> validation -> test split. Proportions
    are computed against the FULL row count first; `purge_bars` is then
    trimmed from the tail of the preceding split (removing training/
    validation samples whose label horizon would otherwise overlap the
    next split) and `embargo_bars + gap_bars` is skipped at the head of the
    following split (an additional serial-correlation buffer, plus any
    extra caller-requested gap) at EACH of the two boundaries."""
    index = pd.DatetimeIndex(pd.Index(timestamps))
    n = len(index)
    if n > 1 and not index.is_monotonic_increasing:
        raise ValidationSplitError("timestamps must be monotonically increasing")
    _validate_fractions(train_fraction, validation_fraction)
    if purge_bars < 0 or embargo_bars < 0 or gap_bars < 0:
        raise ValidationSplitError("purge_bars/embargo_bars/gap_bars must be non-negative")

    raw_train_end = int(n * train_fraction)
    raw_val_end = raw_train_end + int(n * validation_fraction)

    train_stop = raw_train_end - purge_bars
    val_start = raw_train_end + embargo_bars + gap_bars
    val_stop = raw_val_end - purge_bars
    test_start = raw_val_end + embargo_bars + gap_bars

    if train_stop <= 0:
        raise ValidationSplitError(f"Train split is empty after purge_bars={purge_bars} (n={n})")
    if val_stop <= val_start:
        raise ValidationSplitError(
            f"Validation split is empty after purge/embargo/gap (val_start={val_start}, val_stop={val_stop})"
        )
    if test_start >= n:
        raise ValidationSplitError(f"Test split is empty after embargo/gap (test_start={test_start}, n={n})")

    train = DatasetSplit(name="train", indices=np.arange(0, train_stop), start=index[0], end=index[train_stop - 1])
    validation = DatasetSplit(
        name="validation", indices=np.arange(val_start, val_stop), start=index[val_start], end=index[val_stop - 1]
    )
    test = DatasetSplit(name="test", indices=np.arange(test_start, n), start=index[test_start], end=index[n - 1])
    return SplitPlan(
        strategy="chronological", splits=(train, validation, test),
        purge_bars=purge_bars, embargo_bars=embargo_bars, gap_bars=gap_bars,
    )


def build_walk_forward_splits(
    timestamps: pd.Series,
    *,
    n_splits: int,
    test_size: int,
    label_horizon: int = 0,
    embargo: int = 0,
    max_train_size: int | None = None,
) -> SplitPlan:
    """Expanding-window (default, `max_train_size=None`) or rolling-window
    (`max_train_size` set) purged walk-forward folds, via
    `PurgedWalkForwardSplitter`. Each fold contributes a `fold_{k}_train`
    and `fold_{k}_test` `DatasetSplit`."""
    splitter = PurgedWalkForwardSplitter(
        n_splits=n_splits, test_size=test_size, label_horizon=label_horizon, embargo=embargo,
        max_train_size=max_train_size,
    )
    splits: list[DatasetSplit] = []
    for fold in splitter.split(timestamps):
        splits.append(
            DatasetSplit(name=f"fold_{fold.fold}_train", indices=fold.train_indices, start=fold.train_start, end=fold.train_end)
        )
        splits.append(
            DatasetSplit(name=f"fold_{fold.fold}_test", indices=fold.test_indices, start=fold.test_start, end=fold.test_end)
        )
    strategy = "rolling_walk_forward" if max_train_size is not None else "expanding_walk_forward"
    return SplitPlan(strategy=strategy, splits=tuple(splits), purge_bars=label_horizon, embargo_bars=embargo, gap_bars=0)


__all__ = ["DatasetSplit", "SplitPlan", "build_chronological_split", "build_walk_forward_splits"]
