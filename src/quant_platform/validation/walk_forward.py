"""Purged walk-forward cross-validation splitting for time-series ML.

This is a DIFFERENT leakage concern from `TimeframeCursor`'s job.
`TimeframeCursor` prevents a single backtest run from seeing a bar before
it closes (temporal look-ahead within one pass through time). This module
prevents an ML model evaluation from leaking information across the
train/test boundary of a walk-forward split -- a distinct failure mode
that arises from labels that look forward in time (e.g. "did price rise
over the next N bars") and from serial correlation near that boundary.

Purging removes training samples whose label window (`label_horizon` bars
forward) would overlap the test period. Embargo adds a further buffer
immediately before the test period to guard against serial correlation
beyond the label horizon itself. Combined, they establish a single gap of
`label_horizon + embargo` bars between the end of a fold's training data
and the start of its test data. See Lopez de Prado, "Advances in
Financial Machine Learning" (2018), ch. 7, for the underlying methodology.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import ValidationSplitError


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class PurgedWalkForwardSplitter:
    def __init__(
        self,
        n_splits: int,
        test_size: int,
        *,
        label_horizon: int = 0,
        embargo: int = 0,
        max_train_size: int | None = None,
    ) -> None:
        if n_splits <= 0:
            raise ValueError(f"n_splits must be positive, got {n_splits}")
        if test_size <= 0:
            raise ValueError(f"test_size must be positive, got {test_size}")
        if label_horizon < 0:
            raise ValueError(f"label_horizon must be non-negative, got {label_horizon}")
        if embargo < 0:
            raise ValueError(f"embargo must be non-negative, got {embargo}")
        if max_train_size is not None and max_train_size <= 0:
            raise ValueError(f"max_train_size must be positive if given, got {max_train_size}")

        self.n_splits = n_splits
        self.test_size = test_size
        self.label_horizon = label_horizon
        self.embargo = embargo
        self.max_train_size = max_train_size

    @property
    def purge_gap(self) -> int:
        """Total bars kept clear between the end of training data and the
        start of test data in every fold."""
        return self.label_horizon + self.embargo

    def split(self, timestamps: pd.Series | pd.DatetimeIndex) -> Iterator[WalkForwardSplit]:
        index = pd.DatetimeIndex(pd.Index(timestamps))
        n = len(index)

        if n > 1 and not index.is_monotonic_increasing:
            raise ValidationSplitError("timestamps must be monotonically increasing")

        required = self.n_splits * self.test_size
        if required > n:
            raise ValidationSplitError(
                f"Not enough samples ({n}) for {self.n_splits} splits of size "
                f"{self.test_size} each (requires at least {required})"
            )

        gap = self.purge_gap

        for fold in range(self.n_splits):
            test_start = n - (self.n_splits - fold) * self.test_size
            test_end = test_start + self.test_size

            train_end = max(0, test_start - gap)
            train_start = 0 if self.max_train_size is None else max(0, train_end - self.max_train_size)

            if train_start >= train_end:
                raise ValidationSplitError(
                    f"Fold {fold}: no training samples remain after applying the "
                    f"purge/embargo gap ({gap} bars) before test_start index {test_start}"
                )

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            yield WalkForwardSplit(
                fold=fold,
                train_indices=train_indices,
                test_indices=test_indices,
                train_start=index[train_start],
                train_end=index[train_end - 1],
                test_start=index[test_start],
                test_end=index[test_end - 1],
            )
