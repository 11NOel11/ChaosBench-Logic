"""
Integration smoke tests for ChaosBench-Logic evaluation pipeline.

These tests verify that the main evaluation components work together correctly
without making actual API calls. They use manually constructed EvalResult lists
to simulate real evaluation runs.

Tests cover:
- End-to-end summary computation with synthetic data
- Correct metric aggregation across multiple items
- Contradiction and FOL violation detection in realistic scenarios
"""

import pytest
import sys
import os
from dataclasses import dataclass
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eval_chaosbench import compute_summary, EvalResult


def make_result(
    item_id: str,
    pred_norm: Optional[str] = "YES",
    gold: Optional[str] = "YES",
    task_family: str = "atomic",
    dialogue_id: Optional[str] = None,
    turn_index: Optional[int] = None,
    system_id: Optional[str] = "lorenz63",
    question: Optional[str] = None,
    bias_family: Optional[str] = None,
) -> EvalResult:
    """Helper to create synthetic EvalResult for testing."""
    correct = None
    if pred_norm is not None and gold is not None:
        correct = pred_norm == gold

    return EvalResult(
        item_id=item_id,
        batch_file="test_batch.jsonl",
        task_family=task_family,
        bias_family=bias_family,
        dialogue_id=dialogue_id,
        turn_index=turn_index,
        system_id=system_id,
        gold=gold,
        pred_raw=f"Raw output: {pred_norm}",
        pred_norm=pred_norm,
        correct=correct,
        error_type=None,
        question=question,
    )


class TestEndToEndSmoke:
    """Smoke test for complete evaluation flow."""

    def test_small_evaluation_run(self):
        """
        Simulate a small evaluation with 5 items and verify summary structure.
        """
        # Create a small synthetic evaluation with diverse items
        results = [
            # Single atomic question - correct
            make_result("q1", pred_norm="YES", gold="YES", task_family="atomic"),
            # Single multi-hop question - incorrect
            make_result("q2", pred_norm="NO", gold="YES", task_family="multi_hop"),
            # Dialogue with 2 turns - all correct
            make_result(
                "q3",
                pred_norm="YES",
                gold="YES",
                task_family="atomic",
                dialogue_id="d1",
                turn_index=1,
                system_id="lorenz63",
            ),
            make_result(
                "q4",
                pred_norm="NO",
                gold="NO",
                task_family="atomic",
                dialogue_id="d1",
                turn_index=2,
                system_id="lorenz63",
            ),
            # Bias item - incorrect
            make_result(
                "q5",
                pred_norm="YES",
                gold="NO",
                task_family="bias",
                bias_family="chaos_random",
            ),
        ]

        # Run summary computation
        summary = compute_summary(results)

        # Verify all expected keys exist
        assert "overall_accuracy" in summary
        assert "task_accuracy" in summary
        assert "dialogue_accuracy" in summary
        assert "contradiction_rate" in summary
        assert "bias_error" in summary
        assert "avg_violations_per_dialogue" in summary
        assert "violations_breakdown" in summary

        # Verify types
        assert isinstance(summary["overall_accuracy"], float)
        assert isinstance(summary["task_accuracy"], dict)
        assert isinstance(summary["dialogue_accuracy"], float)
        assert isinstance(summary["contradiction_rate"], float)
        assert isinstance(summary["bias_error"], dict)
        assert isinstance(summary["avg_violations_per_dialogue"], (int, float))
        assert isinstance(summary["violations_breakdown"], dict)

        # Verify computed values make sense
        assert 0.0 <= summary["overall_accuracy"] <= 1.0
        assert 0.0 <= summary["dialogue_accuracy"] <= 1.0
        assert 0.0 <= summary["contradiction_rate"] <= 1.0

        # Overall accuracy should be 3/5 = 0.6 (q1, q3, q4 correct)
        assert summary["overall_accuracy"] == pytest.approx(0.6)

        # Task accuracy should have 2 families
        assert "atomic" in summary["task_accuracy"]
        assert "multi_hop" in summary["task_accuracy"]


class TestContradictionDetection:
    """Test that contradictions are properly detected in integration scenario."""

    def test_consistent_dialogue_no_contradictions(self):
        """Consistent dialogue should have 0% contradiction rate."""
        results = [
            make_result(
                "q1",
                pred_norm="YES",
                dialogue_id="d1",
                turn_index=1,
                system_id="lorenz63",
                task_family="atomic",
            ),
            make_result(
                "q2",
                pred_norm="YES",  # Consistent!
                dialogue_id="d1",
                turn_index=2,
                system_id="lorenz63",
                task_family="atomic",
            ),
        ]

        summary = compute_summary(results)
        assert summary["contradiction_rate"] == 0.0

    def test_inconsistent_dialogue_has_contradictions(self):
        """Adding inconsistent answer should increase contradiction rate."""
        # First: consistent dialogue
        results_consistent = [
            make_result(
                "q1",
                pred_norm="YES",
                dialogue_id="d1",
                turn_index=1,
                system_id="lorenz63",
                task_family="atomic",
            ),
            make_result(
                "q2",
                pred_norm="YES",
                dialogue_id="d1",
                turn_index=2,
                system_id="lorenz63",
                task_family="atomic",
            ),
        ]
        summary_consistent = compute_summary(results_consistent)
        assert summary_consistent["contradiction_rate"] == 0.0

        # Second: add contradictory answer
        results_inconsistent = [
            make_result(
                "q1",
                pred_norm="YES",
                dialogue_id="d1",
                turn_index=1,
                system_id="lorenz63",
                task_family="atomic",
            ),
            make_result(
                "q2",
                pred_norm="NO",  # Contradiction!
                dialogue_id="d1",
                turn_index=2,
                system_id="lorenz63",
                task_family="atomic",
            ),
        ]
        summary_inconsistent = compute_summary(results_inconsistent)
        assert summary_inconsistent["contradiction_rate"] == 1.0

        # Verify contradiction rate increased
        assert (
            summary_inconsistent["contradiction_rate"]
            > summary_consistent["contradiction_rate"]
        )


class TestFOLViolationIntegration:
    """Test FOL violation detection in integration scenarios."""

    def test_logically_consistent_dialogue_no_violations(self):
        """Logically consistent predictions should have 0 violations."""
        results = [
            make_result(
                "q1",
                pred_norm="YES",
                dialogue_id="d1",
                turn_index=1,
                system_id="lorenz63",
                question="Is the Lorenz-63 system chaotic?",
            ),
            make_result(
                "q2",
                pred_norm="YES",
                dialogue_id="d1",
                turn_index=2,
                system_id="lorenz63",
                question="Is it deterministic?",
            ),
            make_result(
                "q3",
                pred_norm="YES",
                dialogue_id="d1",
                turn_index=3,
                system_id="lorenz63",
                question="Does it have a positive Lyapunov exponent?",
            ),
        ]

        summary = compute_summary(results)

        # Chaotic=YES, Deterministic=YES, PosLyap=YES is consistent
        assert summary["avg_violations_per_dialogue"] == 0.0
        assert summary["violations_breakdown"]["0_violations"] >= 1

    def test_logically_inconsistent_dialogue_has_violations(self):
        """Adding logically inconsistent answer should increase violations."""
        # First: logically consistent
        results_consistent = [
            make_result(
                "q1",
                pred_norm="YES",
                dialogue_id="d1",
                system_id="lorenz63",
                question="Is the system chaotic?",
            ),
            make_result(
                "q2",
                pred_norm="YES",
                dialogue_id="d1",
                system_id="lorenz63",
                question="Is it deterministic?",
            ),
        ]
        summary_consistent = compute_summary(results_consistent)
        violations_consistent = summary_consistent["avg_violations_per_dialogue"]

        # Second: add logically inconsistent answer
        results_inconsistent = [
            make_result(
                "q1",
                pred_norm="YES",
                dialogue_id="d1",
                system_id="lorenz63",
                question="Is the system chaotic?",
            ),
            make_result(
                "q2",
                pred_norm="NO",  # Inconsistent! Chaotic requires Deterministic
                dialogue_id="d1",
                system_id="lorenz63",
                question="Is it deterministic?",
            ),
        ]
        summary_inconsistent = compute_summary(results_inconsistent)
        violations_inconsistent = summary_inconsistent["avg_violations_per_dialogue"]

        # Verify violations increased
        assert violations_inconsistent > violations_consistent
        assert violations_inconsistent >= 1.0  # At least 1 violation

    def test_violations_breakdown_sums_to_dialogue_count(self):
        """Violations breakdown counts should sum to total dialogues."""
        results = [
            # Dialogue 1: consistent (0 violations)
            make_result(
                "q1",
                pred_norm="YES",
                dialogue_id="d1",
                question="Is it chaotic?",
                system_id="lorenz63",
            ),
            make_result(
                "q2",
                pred_norm="YES",
                dialogue_id="d1",
                question="Is it deterministic?",
                system_id="lorenz63",
            ),
            # Dialogue 2: inconsistent (1+ violations)
            make_result(
                "q3",
                pred_norm="YES",
                dialogue_id="d2",
                question="Is it chaotic?",
                system_id="henon",
            ),
            make_result(
                "q4",
                pred_norm="NO",
                dialogue_id="d2",
                question="Is it deterministic?",
                system_id="henon",
            ),
            # Single question: treated as dialogue with 0 violations
            make_result("q5", pred_norm="YES", question="Is it random?"),
        ]

        summary = compute_summary(results)

        # Sum of breakdown should equal total dialogues (3 in this case)
        breakdown = summary["violations_breakdown"]
        total_dialogues = sum(breakdown.values())
        assert total_dialogues == 3  # d1, d2, and single question q5


class TestBiasErrorIntegration:
    """Test bias error computation in integration scenarios."""

    def test_bias_errors_computed_correctly(self):
        """Bias items should be correctly counted in bias_error metric."""
        results = [
            # chaos_random bias: 1 correct, 1 incorrect
            make_result(
                "q1", pred_norm="YES", gold="YES", bias_family="chaos_random"
            ),
            make_result(
                "q2", pred_norm="NO", gold="YES", bias_family="chaos_random"
            ),  # Error
            # determ_random bias: 2 correct
            make_result(
                "q3", pred_norm="NO", gold="NO", bias_family="determ_random"
            ),
            make_result(
                "q4", pred_norm="YES", gold="YES", bias_family="determ_random"
            ),
        ]

        summary = compute_summary(results)

        assert "chaos_random" in summary["bias_error"]
        assert "determ_random" in summary["bias_error"]

        # chaos_random: 1 error out of 2 = 0.5
        assert summary["bias_error"]["chaos_random"] == 0.5

        # determ_random: 0 errors out of 2 = 0.0
        assert summary["bias_error"]["determ_random"] == 0.0


# Summary test
def test_integration_smoke_coverage():
    """
    Meta-test documenting integration smoke test coverage.

    Coverage:
    ✓ End-to-end evaluation flow (1 test)
    ✓ Contradiction detection (2 tests)
    ✓ FOL violation integration (3 tests)
    ✓ Bias error integration (1 test)

    Total: 7 integration smoke tests
    These verify that all components work together correctly.
    """
    pass
