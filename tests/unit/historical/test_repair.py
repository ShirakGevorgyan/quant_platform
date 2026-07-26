"""Tests for `historical.repair.apply_repair_policy`."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone as dt_timezone

import pandas as pd
import pytest

from quant_platform.core.exceptions import QuarantineError
from quant_platform.core.types import Timeframe
from quant_platform.data.synthetic import SyntheticDataConfig, generate_ohlcv
from quant_platform.historical.quality import run_quality_checks
from quant_platform.historical.repair import RepairAction, SeverityPolicy, apply_repair_policy


def _frame(n: int = 20, seed: int = 1) -> pd.DataFrame:
    sd = generate_ohlcv(
        SyntheticDataConfig(start=datetime(2024, 1, 3, tzinfo=dt_timezone.utc), periods=n, timeframe=Timeframe.M1, seed=seed)
    )
    sd = sd.rename(columns={"volume": "tick_volume"})
    sd["real_volume"] = 0
    sd["spread"] = 15
    return sd


class TestExactDuplicateRemoval:
    def test_byte_identical_duplicate_is_removed_regardless_of_policy(self) -> None:
        df = _frame()
        df.loc[5] = df.loc[4]  # byte-identical copy, including open_time
        report = run_quality_checks(df, symbol="X", timeframe=Timeframe.M1)
        result = apply_repair_policy(df, report, policy=SeverityPolicy.STRICT)
        assert result.lineage.rows_out == len(df) - 1
        assert result.lineage.rows_quarantined == 0
        step = result.lineage.steps[0]
        assert step.action is RepairAction.EXACT_DUPLICATE_REMOVED
        assert step.affected_row_indices == (5,)

    def test_disabling_exact_duplicate_removal_leaves_it_for_policy_to_handle(self) -> None:
        df = _frame()
        df.loc[5] = df.loc[4]
        report = run_quality_checks(df, symbol="X", timeframe=Timeframe.M1)
        with pytest.raises(QuarantineError):
            apply_repair_policy(df, report, policy=SeverityPolicy.STRICT, allow_exact_duplicate_removal=False)


class TestConflictingDuplicates:
    """A conflicting duplicate (same open_time, different OHLC) is NOT an
    exact duplicate -- it must be resolved through the explicit severity
    policy, never silently."""

    def _conflicting_frame(self) -> pd.DataFrame:
        df = _frame()
        df.loc[6, "open_time"] = df.loc[5, "open_time"]
        return df

    def test_strict_policy_rejects_the_whole_batch(self) -> None:
        df = self._conflicting_frame()
        report = run_quality_checks(df, symbol="X", timeframe=Timeframe.M1)
        with pytest.raises(QuarantineError, match="critical quality issue"):
            apply_repair_policy(df, report, policy=SeverityPolicy.STRICT)

    def test_quarantine_policy_isolates_exactly_the_conflicting_row(self) -> None:
        df = self._conflicting_frame()
        report = run_quality_checks(df, symbol="X", timeframe=Timeframe.M1)
        result = apply_repair_policy(df, report, policy=SeverityPolicy.QUARANTINE)
        assert result.lineage.rows_out == len(df) - 1
        assert result.lineage.rows_quarantined == 1
        # The FIRST occurrence (row 5) is kept; the later one (row 6) is quarantined.
        assert result.quarantined["open_time"].iloc[0] == df["open_time"].iloc[6]
        assert (result.data["open_time"] == df["open_time"].iloc[5]).sum() == 1

    def test_warn_only_policy_retains_every_row(self) -> None:
        df = self._conflicting_frame()
        report = run_quality_checks(df, symbol="X", timeframe=Timeframe.M1)
        result = apply_repair_policy(df, report, policy=SeverityPolicy.WARN_ONLY)
        assert result.lineage.rows_out == len(df)
        assert result.lineage.rows_quarantined == 0
        assert result.lineage.steps[-1].action is RepairAction.RETAINED_WITH_WARNING

    def test_no_prices_are_ever_interpolated_or_synthesized(self) -> None:
        df = self._conflicting_frame()
        report = run_quality_checks(df, symbol="X", timeframe=Timeframe.M1)
        result = apply_repair_policy(df, report, policy=SeverityPolicy.QUARANTINE)
        # Every surviving row's OHLC values must be byte-identical to some
        # row in the original input -- nothing computed/interpolated.
        original_tuples = {tuple(row) for row in df[["open", "high", "low", "close"]].itertuples(index=False)}
        for row in result.data[["open", "high", "low", "close"]].itertuples(index=False):
            assert tuple(row) in original_tuples


class TestSorting:
    def test_sort_is_a_noop_unless_explicitly_allowed(self) -> None:
        df = _frame()
        shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
        report = run_quality_checks(shuffled, symbol="X", timeframe=Timeframe.M1)
        result = apply_repair_policy(shuffled, report, policy=SeverityPolicy.WARN_ONLY, allow_sort=False)
        assert not any(s.action is RepairAction.SORTED for s in result.lineage.steps)
        pd.testing.assert_frame_equal(result.data.reset_index(drop=True), shuffled.reset_index(drop=True))

    def test_sort_is_applied_and_reported_when_allowed(self) -> None:
        df = _frame()
        shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
        report = run_quality_checks(shuffled, symbol="X", timeframe=Timeframe.M1)
        result = apply_repair_policy(shuffled, report, policy=SeverityPolicy.WARN_ONLY, allow_sort=True)
        assert result.data["open_time"].is_monotonic_increasing
        assert any(s.action is RepairAction.SORTED for s in result.lineage.steps)


class TestLineageCompleteness:
    def test_clean_data_produces_empty_steps_but_full_lineage(self) -> None:
        df = _frame()
        report = run_quality_checks(df, symbol="X", timeframe=Timeframe.M1)
        result = apply_repair_policy(df, report, policy=SeverityPolicy.STRICT, input_snapshot_id="snap-123")
        assert result.lineage.steps == ()
        assert result.lineage.rows_in == result.lineage.rows_out == len(df)
        assert result.lineage.input_snapshot_id == "snap-123"
        assert result.lineage.resulting_checksum
        assert result.lineage.transformation_name
        assert result.lineage.transformation_version
        assert result.lineage.parameters["policy"] == "STRICT"

    def test_checksum_is_deterministic_for_identical_output(self) -> None:
        df = _frame()
        report = run_quality_checks(df, symbol="X", timeframe=Timeframe.M1)
        result_a = apply_repair_policy(df, report, policy=SeverityPolicy.STRICT)
        result_b = apply_repair_policy(df, report, policy=SeverityPolicy.STRICT)
        assert result_a.lineage.resulting_checksum == result_b.lineage.resulting_checksum

    def test_checksum_changes_when_output_changes(self) -> None:
        df = _frame()
        df.loc[5] = df.loc[4]
        report = run_quality_checks(df, symbol="X", timeframe=Timeframe.M1)
        result = apply_repair_policy(df, report, policy=SeverityPolicy.STRICT)
        clean_report = run_quality_checks(_frame(), symbol="X", timeframe=Timeframe.M1)
        clean_result = apply_repair_policy(_frame(), clean_report, policy=SeverityPolicy.STRICT)
        assert result.lineage.resulting_checksum != clean_result.lineage.resulting_checksum


class TestEmptyInput:
    def test_empty_dataframe_is_handled_without_error(self) -> None:
        df = _frame(1).iloc[0:0]
        report = run_quality_checks(df, symbol="X", timeframe=Timeframe.M1)
        result = apply_repair_policy(df, report, policy=SeverityPolicy.STRICT)
        assert result.lineage.rows_in == 0
        assert result.lineage.rows_out == 0
        assert result.lineage.rows_quarantined == 0
