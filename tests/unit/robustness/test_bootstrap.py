"""Milestone 6, Section 16: deterministic bootstrap, block-bootstrap
boundary behavior, confidence intervals, and undefined-sample handling.
Covers the exact regression this session found and fixed in
`_stationary_resample` (every earlier "resample" degenerating into a
circular rotation of the same series, collapsing the CI to a near-zero
width) via a direct width comparison against `moving_block`."""

from __future__ import annotations

import pytest

from quant_platform.core.exceptions import BootstrapError
from quant_platform.robustness.bootstrap import (
    STANDARD_STATISTICS,
    _stationary_resample,
    bootstrap_statistic,
    compute_bootstrap_report,
    make_profit_factor_statistic,
    make_total_return_statistic,
)
from quant_platform.robustness.models import BootstrapMethodKind, ReturnSeriesKind
from quant_platform.robustness.series import FoldBoundary, ReturnSeriesBundle
from quant_platform.robustness.specs import BootstrapSpec


def _bundle(values: tuple[float, ...], *, fold_boundaries: tuple[FoldBoundary, ...] = ()) -> ReturnSeriesBundle:
    return ReturnSeriesBundle(
        schema_version=1, kind=ReturnSeriesKind.STITCHED_BAR_NET, sampling_frequency="bar", observation_count=len(values),
        effective_sample_count=len(values), values=values, fold_boundaries=fold_boundaries, source_artifact_content_hashes=("a" * 64,),
        time_range_start=None, time_range_end=None, built_at="2026-01-01T00:00:00Z",
    )


_SYNTHETIC_SERIES = tuple(0.01 * ((i % 7) - 3) for i in range(60))  # deterministic, mixed-sign, autocorrelation-free-ish


class TestDeterministicSeeding:
    def test_same_seed_produces_bit_identical_estimate(self) -> None:
        bundle = _bundle(_SYNTHETIC_SERIES)
        spec = BootstrapSpec(method=BootstrapMethodKind.STATIONARY, repetitions=200, confidence_level=0.95, block_length=5)
        a = bootstrap_statistic(bundle, statistic_name="total_return", statistic_fn=make_total_return_statistic(), spec=spec, seed=42)
        b = bootstrap_statistic(bundle, statistic_name="total_return", statistic_fn=make_total_return_statistic(), spec=spec, seed=42)
        assert a == b

    def test_different_seed_produces_different_bounds(self) -> None:
        bundle = _bundle(_SYNTHETIC_SERIES)
        spec = BootstrapSpec(method=BootstrapMethodKind.STATIONARY, repetitions=200, confidence_level=0.95, block_length=5)
        a = bootstrap_statistic(bundle, statistic_name="total_return", statistic_fn=make_total_return_statistic(), spec=spec, seed=1)
        b = bootstrap_statistic(bundle, statistic_name="total_return", statistic_fn=make_total_return_statistic(), spec=spec, seed=2)
        assert (a.lower_bound, a.upper_bound) != (b.lower_bound, b.upper_bound)


class TestBlockBootstrapBoundaryBehavior:
    def test_moving_block_with_block_length_equal_to_series_length_collapses_to_a_point(self) -> None:
        """Hand-verifiable: with `block_length == n`, there is exactly ONE
        valid starting position (`n - block_length + 1 == 1`), so every
        resample is the ORIGINAL series, unchanged -- the CI must collapse
        exactly to the point estimate, not merely be narrow."""
        values = (0.01, -0.02, 0.03, 0.015, -0.01)
        bundle = _bundle(values)
        spec = BootstrapSpec(method=BootstrapMethodKind.MOVING_BLOCK, repetitions=200, confidence_level=0.95, block_length=len(values))
        estimate = bootstrap_statistic(bundle, statistic_name="total_return", statistic_fn=make_total_return_statistic(), spec=spec, seed=7)
        expected_total_return = 1.0
        for v in values:
            expected_total_return *= 1.0 + v
        expected_total_return -= 1.0
        assert estimate.point_estimate == pytest.approx(expected_total_return, abs=1e-12)
        assert estimate.lower_bound == pytest.approx(expected_total_return, abs=1e-12)
        assert estimate.upper_bound == pytest.approx(expected_total_return, abs=1e-12)
        assert estimate.valid_repetitions == 200

    def test_fold_level_with_a_single_fold_collapses_to_a_point(self) -> None:
        """With exactly one declared fold, `_fold_level_resample` has
        only one possible choice every repetition -- hand-verifiable
        degenerate case identical in spirit to the moving-block one
        above, for the FOLD_LEVEL method instead."""
        values = (0.01, -0.02, 0.03, 0.015, -0.01)
        bundle = _bundle(values, fold_boundaries=(FoldBoundary(outer_fold_index=0, start_index=0, end_index=len(values) - 1),))
        spec = BootstrapSpec(method=BootstrapMethodKind.FOLD_LEVEL, repetitions=100, confidence_level=0.9)
        estimate = bootstrap_statistic(bundle, statistic_name="total_return", statistic_fn=make_total_return_statistic(), spec=spec, seed=3)
        assert estimate.lower_bound == pytest.approx(estimate.upper_bound, abs=1e-12)
        assert estimate.point_estimate == pytest.approx(estimate.lower_bound, abs=1e-12)

    def test_stationary_bootstrap_is_not_degenerate_unlike_the_regression_this_session_fixed(self) -> None:
        """Regression test for the exact bug found and fixed in
        `_stationary_resample`: an earlier version only randomized the
        first block's start position and then let it drift forward
        contiguously, degenerating into a pure circular rotation of the
        series -- order-independent statistics like total_return were
        then IDENTICAL across every "resample", collapsing the CI to a
        near-zero width. Asserts the stationary CI width is within the
        same order of magnitude as `moving_block`'s own CI width on the
        identical series -- a collapsed CI would be orders of magnitude
        narrower."""
        bundle = _bundle(_SYNTHETIC_SERIES)
        stationary_spec = BootstrapSpec(method=BootstrapMethodKind.STATIONARY, repetitions=500, confidence_level=0.95, block_length=5)
        moving_block_spec = BootstrapSpec(method=BootstrapMethodKind.MOVING_BLOCK, repetitions=500, confidence_level=0.95, block_length=5)
        stationary = bootstrap_statistic(bundle, statistic_name="total_return", statistic_fn=make_total_return_statistic(), spec=stationary_spec, seed=11)
        moving_block = bootstrap_statistic(bundle, statistic_name="total_return", statistic_fn=make_total_return_statistic(), spec=moving_block_spec, seed=11)
        stationary_width = stationary.upper_bound - stationary.lower_bound
        moving_block_width = moving_block.upper_bound - moving_block.lower_bound
        assert stationary_width > 0.0, "stationary bootstrap CI collapsed to a point -- the exact regression this test guards against"
        assert stationary_width > 0.1 * moving_block_width


class TestStationaryResampleMutationDetection:
    """Release-audit Section 7 (flagged for special strictness since a
    degenerate-rotation bug was already found once in `_stationary_
    resample`): reintroduces each specific known-bad implementation
    variant and confirms the CURRENT implementation is statistically
    distinguishable from it, via the number of DISTINCT `total_return`
    values produced across many independently-seeded resamples of the
    same series. A degenerate "single rotation" resampler can produce at
    most `n` distinct total_return values (one per starting rotation
    offset); the correct stationary bootstrap, which independently
    randomizes EVERY block's start and length, produces far more."""

    @staticmethod
    def _distinct_total_return_count(resample_fn, *, values, n_seeds: int) -> int:
        import numpy as np

        seen: set[float] = set()
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            resample = resample_fn(values, rng=rng)
            product = 1.0
            for v in resample:
                product *= 1.0 + float(v)
            seen.add(round(product - 1.0, 10))
        return len(seen)

    def test_random_first_block_only_then_drift_forever_is_detected(self) -> None:
        """The EXACT original bug: only the first block's start is
        randomized; `pos` then drifts forward contiguously forever,
        degenerating into a single circular rotation of the whole
        series."""
        import numpy as np

        def buggy_resample(values: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
            n = len(values)
            pos = int(rng.integers(0, n))
            out = [float(values[(pos + i) % n]) for i in range(n)]
            return np.asarray(out)

        values = np.array([0.01 * ((i % 7) - 3) for i in range(30)])
        buggy_distinct = self._distinct_total_return_count(buggy_resample, values=values, n_seeds=200)
        correct_distinct = self._distinct_total_return_count(
            lambda v, *, rng: _stationary_resample(v, mean_block_length=5, rng=rng), values=values, n_seeds=200,
        )
        assert buggy_distinct <= len(values), "sanity: a single rotation can produce at most n distinct values"
        assert correct_distinct > buggy_distinct, "current implementation must be detectably richer than a single-rotation degenerate resampler"

    def test_no_restart_between_blocks_is_detected(self) -> None:
        """A variant that picks ONE random block length and ONE random
        start for the WHOLE resample (never restarting per-block) is the
        same degenerate rotation as above, just parameterized differently."""
        import numpy as np

        def buggy_resample(values: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
            n = len(values)
            pos = int(rng.integers(0, n))
            return np.asarray([float(values[(pos + i) % n]) for i in range(n)])

        values = np.array([0.01 * ((i % 5) - 2) for i in range(24)])
        buggy_distinct = self._distinct_total_return_count(buggy_resample, values=values, n_seeds=200)
        correct_distinct = self._distinct_total_return_count(
            lambda v, *, rng: _stationary_resample(v, mean_block_length=4, rng=rng), values=values, n_seeds=200,
        )
        assert correct_distinct > buggy_distinct

    def test_no_wraparound_truncates_and_is_detected(self) -> None:
        """A variant that does NOT wrap `pos` modulo `n` will either index
        out of bounds or (if clipped) systematically under-sample the
        series' tail -- confirmed distinguishable via the resample length
        distribution (the correct implementation always returns exactly
        `n` values; a naive non-wrapping port would either crash or
        require truncation logic that changes the result distribution)."""
        import numpy as np

        rng = np.random.default_rng(0)
        values = np.arange(20, dtype=float)
        resample = _stationary_resample(values, mean_block_length=6, rng=rng)
        assert len(resample) == len(values)
        # A resample containing a value equal to n-1 immediately followed by
        # 0 (wraparound) somewhere is expected with reasonable probability
        # over many seeds -- a non-wrapping implementation could never
        # produce this adjacency.
        saw_wraparound_adjacency = False
        for seed in range(50):
            r = _stationary_resample(values, mean_block_length=6, rng=np.random.default_rng(seed))
            for i in range(len(r) - 1):
                if r[i] == len(values) - 1 and r[i + 1] == 0.0:
                    saw_wraparound_adjacency = True
                    break
            if saw_wraparound_adjacency:
                break
        assert saw_wraparound_adjacency, "expected at least one wraparound (n-1 -> 0) adjacency across 50 independent seeds"

    def test_deterministic_restart_position_is_detected(self) -> None:
        """A variant that always restarts each new block at a FIXED
        position (e.g. always 0) rather than a fresh random one collapses
        to a small, highly repetitive set of possible resamples."""
        import numpy as np

        def buggy_resample(values: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
            n = len(values)
            p = 1.0 / 5
            out: list[float] = []
            while len(out) < n:
                length = int(rng.geometric(p))
                for i in range(length):
                    if len(out) >= n:
                        break
                    out.append(float(values[i % n]))  # ALWAYS restarts at position 0, never randomized
            return np.asarray(out[:n])

        values = np.array([0.01 * ((i % 7) - 3) for i in range(30)])
        buggy_distinct = self._distinct_total_return_count(buggy_resample, values=values, n_seeds=200)
        correct_distinct = self._distinct_total_return_count(
            lambda v, *, rng: _stationary_resample(v, mean_block_length=5, rng=rng), values=values, n_seeds=200,
        )
        assert correct_distinct > buggy_distinct


class TestUndefinedSamplesNeverSilentlyCoerced:
    def test_profit_factor_undefined_for_every_resample_of_an_all_positive_series_fails_closed(self) -> None:
        """A strictly-positive series has zero losing periods in EVERY
        possible resample -- profit_factor is undefined for all of them.
        `bootstrap_statistic` must raise, never report `0.0`/`inf`/`NaN`."""
        bundle = _bundle(tuple(0.01 for _ in range(20)))
        spec = BootstrapSpec(method=BootstrapMethodKind.IID, repetitions=200, confidence_level=0.95)
        with pytest.raises(BootstrapError):
            bootstrap_statistic(bundle, statistic_name="profit_factor", statistic_fn=make_profit_factor_statistic(), spec=spec, seed=5)

    def test_partial_undefined_repetitions_are_counted_and_excluded_not_coerced(self) -> None:
        """A series with exactly one negative value among mostly-positive
        ones: some IID resamples will happen to avoid drawing the
        negative value at all (profit_factor undefined for those), others
        will include it (profit_factor defined). Both valid and skipped
        counts must be > 0, and the reported bounds must be finite."""
        values = (0.1, 0.1, 0.1, 0.1, -0.05)
        bundle = _bundle(values)
        spec = BootstrapSpec(method=BootstrapMethodKind.IID, repetitions=500, confidence_level=0.9)
        estimate = bootstrap_statistic(bundle, statistic_name="profit_factor", statistic_fn=make_profit_factor_statistic(), spec=spec, seed=13)
        assert estimate.valid_repetitions > 0
        assert estimate.skipped_repetitions > 0
        assert estimate.valid_repetitions + estimate.skipped_repetitions == 500
        assert estimate.point_estimate is not None
        import math

        assert math.isfinite(estimate.point_estimate)
        assert math.isfinite(estimate.lower_bound) and math.isfinite(estimate.upper_bound)
        assert sum(estimate.failure_reasons.values()) == estimate.skipped_repetitions


class TestComputeBootstrapReport:
    def test_default_report_covers_every_standard_statistic(self) -> None:
        bundle = _bundle(_SYNTHETIC_SERIES)
        spec = BootstrapSpec(method=BootstrapMethodKind.IID, repetitions=200, confidence_level=0.95)
        report = compute_bootstrap_report(bundle, spec=spec, seed=1)
        assert {e.statistic_name for e in report.estimates} == set(STANDARD_STATISTICS)

    def test_json_round_trip(self) -> None:
        bundle = _bundle(_SYNTHETIC_SERIES)
        spec = BootstrapSpec(method=BootstrapMethodKind.IID, repetitions=150, confidence_level=0.9)
        report = compute_bootstrap_report(bundle, spec=spec, seed=2)
        roundtripped = type(report).from_json_dict(report.to_json_dict())
        assert roundtripped == report
