from __future__ import annotations

import pytest

from quant_platform.execution.state_machine import (
    TERMINAL_STAGES,
    ExecutionStage,
    is_legal_execution_transition,
    is_terminal_stage,
)


class TestLegalTransitions:
    @pytest.mark.parametrize(
        ("current", "target"),
        [
            (ExecutionStage.INITIALIZING, ExecutionStage.LOADING_DATASET),
            (ExecutionStage.LOADING_DATASET, ExecutionStage.BUILDING_SPLITS),
            (ExecutionStage.BUILDING_SPLITS, ExecutionStage.RUNNING_FOLD),
            (ExecutionStage.RUNNING_FOLD, ExecutionStage.STORING_RESULTS),
            (ExecutionStage.RUNNING_FOLD, ExecutionStage.RECOVERABLE_FAILURE),
            (ExecutionStage.STORING_RESULTS, ExecutionStage.RUNNING_FOLD),
            (ExecutionStage.STORING_RESULTS, ExecutionStage.COMPLETED),
            (ExecutionStage.RECOVERABLE_FAILURE, ExecutionStage.RUNNING_FOLD),
        ],
    )
    def test_expected_forward_transitions_are_legal(self, current: ExecutionStage, target: ExecutionStage) -> None:
        assert is_legal_execution_transition(current, target)

    @pytest.mark.parametrize("stage", [ExecutionStage.INITIALIZING, ExecutionStage.LOADING_DATASET, ExecutionStage.BUILDING_SPLITS, ExecutionStage.RUNNING_FOLD, ExecutionStage.STORING_RESULTS, ExecutionStage.RECOVERABLE_FAILURE])
    def test_every_non_terminal_stage_can_reach_failed_and_cancelled(self, stage: ExecutionStage) -> None:
        assert is_legal_execution_transition(stage, ExecutionStage.FAILED)
        assert is_legal_execution_transition(stage, ExecutionStage.CANCELLED)

    @pytest.mark.parametrize("terminal", [ExecutionStage.COMPLETED, ExecutionStage.FAILED, ExecutionStage.CANCELLED])
    def test_no_transition_out_of_any_terminal_stage(self, terminal: ExecutionStage) -> None:
        for target in ExecutionStage:
            assert not is_legal_execution_transition(terminal, target)

    def test_no_meaningless_self_loops_exist_in_the_table(self) -> None:
        """`ExecutionManifestStore.bump_resume_count` exists precisely
        BECAUSE no stage is allowed to transition to itself -- this test
        pins that design choice so a future edit doesn't silently add a
        self-loop that would make `bump_resume_count` redundant without
        anyone noticing."""
        for stage in ExecutionStage:
            assert not is_legal_execution_transition(stage, stage)

    def test_running_fold_cannot_go_directly_to_completed(self) -> None:
        """Pins the invariant `execution.runner.ExecutionRunner` relies
        on: the terminal transition is only legal FROM `STORING_RESULTS`,
        never directly from `RUNNING_FOLD` -- which is why the runner
        explicitly passes through `STORING_RESULTS` even when a resumed
        run has nothing left to do."""
        assert not is_legal_execution_transition(ExecutionStage.RUNNING_FOLD, ExecutionStage.COMPLETED)


class TestTerminalStageHelpers:
    @pytest.mark.parametrize("stage", [ExecutionStage.COMPLETED, ExecutionStage.FAILED, ExecutionStage.CANCELLED])
    def test_is_terminal_stage_true_for_terminal(self, stage: ExecutionStage) -> None:
        assert is_terminal_stage(stage)
        assert stage in TERMINAL_STAGES

    @pytest.mark.parametrize(
        "stage",
        [
            ExecutionStage.INITIALIZING, ExecutionStage.LOADING_DATASET, ExecutionStage.BUILDING_SPLITS,
            ExecutionStage.RUNNING_FOLD, ExecutionStage.STORING_RESULTS, ExecutionStage.RECOVERABLE_FAILURE,
        ],
    )
    def test_is_terminal_stage_false_for_non_terminal(self, stage: ExecutionStage) -> None:
        assert not is_terminal_stage(stage)
        assert stage not in TERMINAL_STAGES

    def test_every_enum_member_is_covered_by_the_transition_table(self) -> None:
        from quant_platform.execution.state_machine import _LEGAL_TRANSITIONS

        assert set(_LEGAL_TRANSITIONS) == set(ExecutionStage)
