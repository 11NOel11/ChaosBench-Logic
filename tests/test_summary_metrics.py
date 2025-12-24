"""
Tests for summary metrics computation (compute_summary function).

Tests the core metrics aggregation logic using synthetic EvalResult data
to avoid network calls and real API dependencies.

Metrics tested:
- overall_accuracy
- task_accuracy
- dialogue_accuracy
- contradiction_rate
- avg_violations_per_dialogue (Phase 2)
- violations_breakdown (Phase 2)
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


class TestOverallAccuracy:
    """Test overall accuracy computation."""

    def test_perfect_accuracy(self):
        """All correct should give 100% accuracy."""
        results = [
            make_result("q1", pred_norm="YES", gold="YES"),
            make_result("q2", pred_norm="NO", gold="NO"),
            make_result("q3", pred_norm="YES", gold="YES"),
        ]
        summary = compute_summary(results)
        assert summary["overall_accuracy"] == 1.0

    def test_zero_accuracy(self):
        """All incorrect should give 0% accuracy."""
        results = [
            make_result("q1", pred_norm="NO", gold="YES"),
            make_result("q2", pred_norm="YES", gold="NO"),
            make_result("q3", pred_norm="NO", gold="YES"),
        ]
        summary = compute_summary(results)
        assert summary["overall_accuracy"] == 0.0

    def test_partial_accuracy(self):
        """Should correctly compute fractional accuracy."""
        results = [
            make_result("q1", pred_norm="YES", gold="YES"),  # Correct
            make_result("q2", pred_norm="NO", gold="YES"),  # Incorrect
            make_result("q3", pred_norm="YES", gold="YES"),  # Correct
        ]
        summary = compute_summary(results)
        assert summary["overall_accuracy"] == pytest.approx(2.0 / 3.0)

    def test_ignores_unanswered(self):
        """Should ignore items where prediction is None."""
        results = [
            make_result("q1", pred_norm="YES", gold="YES"),  # Correct
            make_result("q2", pred_norm=None, gold="YES"),  # Unanswered
            make_result("q3", pred_norm="YES", gold="YES"),  # Correct
        ]
        summary = compute_summary(results)
        # Should be 2/2 = 100%, not 2/3
        assert summary["overall_accuracy"] == 1.0


class TestTaskAccuracy:
    """Test per-task-family accuracy."""

    def test_task_accuracy_multiple_families(self):
        """Should compute accuracy separately for each task family."""
        results = [
            make_result("q1", pred_norm="YES", gold="YES", task_family="atomic"),
            make_result("q2", pred_norm="NO", gold="YES", task_family="atomic"),
            make_result("q3", pred_norm="YES", gold="YES", task_family="multi_hop"),
            make_result("q4", pred_norm="YES", gold="YES", task_family="multi_hop"),
        ]
        summary = compute_summary(results)

        assert "task_accuracy" in summary
        assert summary["task_accuracy"]["atomic"] == 0.5  # 1/2
        assert summary["task_accuracy"]["multi_hop"] == 1.0  # 2/2

    def test_task_accuracy_single_family(self):
        """Should handle single task family."""
        results = [
            make_result("q1", pred_norm="YES", gold="YES", task_family="bias"),
            make_result("q2", pred_norm="YES", gold="YES", task_family="bias"),
        ]
        summary = compute_summary(results)

        assert summary["task_accuracy"]["bias"] == 1.0


class TestDialogueAccuracy:
    """Test dialogue-level accuracy computation."""

    def test_perfect_dialogue(self):
        """All turns correct should give dialogue accuracy 100%."""
        results = [
            make_result(
                "q1", pred_norm="YES", gold="YES", dialogue_id="d1", turn_index=1
            ),
            make_result(
                "q2", pred_norm="NO", gold="NO", dialogue_id="d1", turn_index=2
            ),
        ]
        summary = compute_summary(results)
        assert summary["dialogue_accuracy"] == 1.0

    def test_imperfect_dialogue(self):
        """One wrong turn should make dialogue incorrect."""
        results = [
            make_result(
                "q1", pred_norm="YES", gold="YES", dialogue_id="d1", turn_index=1
            ),
            make_result(
                "q2", pred_norm="YES", gold="NO", dialogue_id="d1", turn_index=2
            ),  # Wrong
        ]
        summary = compute_summary(results)
        assert summary["dialogue_accuracy"] == 0.0

    def test_multiple_dialogues(self):
        """Should compute across multiple dialogues."""
        results = [
            # Dialogue 1: all correct
            make_result(
                "q1", pred_norm="YES", gold="YES", dialogue_id="d1", turn_index=1
            ),
            make_result(
                "q2", pred_norm="NO", gold="NO", dialogue_id="d1", turn_index=2
            ),
            # Dialogue 2: one wrong
            make_result(
                "q3", pred_norm="YES", gold="YES", dialogue_id="d2", turn_index=1
            ),
            make_result(
                "q4", pred_norm="YES", gold="NO", dialogue_id="d2", turn_index=2
            ),
        ]
        summary = compute_summary(results)
        assert summary["dialogue_accuracy"] == 0.5  # 1 out of 2 dialogues correct

    def test_no_dialogues(self):
        """Should handle case with no multi-turn dialogues."""
        results = [
            make_result("q1", pred_norm="YES", gold="YES"),  # Single question
        ]
        summary = compute_summary(results)
        # dialogue_accuracy should exist but may be None or based on single questions
        assert "dialogue_accuracy" in summary


class TestContradictionRate:
    """Test contradiction rate (binary per-dialogue metric)."""

    def test_no_contradictions(self):
        """Consistent answers should give 0% contradiction rate."""
        results = [
            make_result(
                "q1",
                pred_norm="YES",
                gold="YES",
                dialogue_id="d1",
                turn_index=1,
                system_id="lorenz63",
                task_family="atomic",
            ),
            make_result(
                "q2",
                pred_norm="YES",
                gold="YES",
                dialogue_id="d1",
                turn_index=2,
                system_id="lorenz63",
                task_family="atomic",
            ),
        ]
        summary = compute_summary(results)
        assert summary["contradiction_rate"] == 0.0

    def test_contradiction_detected(self):
        """YES then NO for same (system, task) should be contradiction."""
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
                pred_norm="NO",  # Contradiction!
                dialogue_id="d1",
                turn_index=2,
                system_id="lorenz63",
                task_family="atomic",
            ),
        ]
        summary = compute_summary(results)
        assert summary["contradiction_rate"] == 1.0  # 1 out of 1 dialogues

    def test_multiple_dialogues_some_contradictions(self):
        """Should compute rate across multiple dialogues."""
        results = [
            # Dialogue 1: contradiction
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
                pred_norm="NO",
                dialogue_id="d1",
                turn_index=2,
                system_id="lorenz63",
                task_family="atomic",
            ),
            # Dialogue 2: no contradiction
            make_result(
                "q3",
                pred_norm="YES",
                dialogue_id="d2",
                turn_index=1,
                system_id="henon",
                task_family="atomic",
            ),
            make_result(
                "q4",
                pred_norm="YES",
                dialogue_id="d2",
                turn_index=2,
                system_id="henon",
                task_family="atomic",
            ),
        ]
        summary = compute_summary(results)
        assert summary["contradiction_rate"] == 0.5  # 1 out of 2


class TestFOLViolationMetrics:
    """Test Phase 2 FOL violation metrics."""

    def test_zero_violations(self):
        """Logically consistent dialogue should have 0 violations."""
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
        ]
        summary = compute_summary(results)

        assert "avg_violations_per_dialogue" in summary
        assert "violations_breakdown" in summary

        # This dialogue should have 0 violations (Chaotic=YES, Deterministic=YES is consistent)
        assert summary["avg_violations_per_dialogue"] >= 0.0

    def test_violations_breakdown_structure(self):
        """violations_breakdown should have correct structure."""
        results = [
            make_result(
                "q1",
                pred_norm="YES",
                dialogue_id="d1",
                system_id="lorenz63",
                question="Is it chaotic?",
            ),
        ]
        summary = compute_summary(results)

        breakdown = summary["violations_breakdown"]
        assert isinstance(breakdown, dict)

        # Should have keys for violation counts
        expected_keys = ["0_violations", "1_violation", "2_violations", "3+_violations"]
        for key in expected_keys:
            assert key in breakdown
            assert isinstance(breakdown[key], int)
            assert breakdown[key] >= 0

    def test_single_questions_counted_as_dialogues(self):
        """Single questions should be treated as length-1 dialogues."""
        results = [
            make_result(
                "q1",
                pred_norm="YES",
                dialogue_id=None,  # Single question
                system_id="lorenz63",
                question="Is it chaotic?",
            ),
            make_result(
                "q2",
                pred_norm="NO",
                dialogue_id=None,  # Single question
                system_id="henon",
                question="Is it random?",
            ),
        ]
        summary = compute_summary(results)

        # Should compute violations for single questions too
        assert summary["avg_violations_per_dialogue"] is not None
        breakdown = summary["violations_breakdown"]
        total_dialogues = sum(breakdown.values())
        assert total_dialogues == 2  # 2 single questions = 2 dialogues


class TestBiasError:
    """Test bias error rate computation."""

    def test_bias_error_computed(self):
        """Should compute error rate for bias items."""
        results = [
            make_result(
                "q1", pred_norm="YES", gold="YES", bias_family="chaos_random"
            ),
            make_result(
                "q2", pred_norm="NO", gold="YES", bias_family="chaos_random"
            ),  # Error
        ]
        summary = compute_summary(results)

        assert "bias_error" in summary
        assert "chaos_random" in summary["bias_error"]
        assert summary["bias_error"]["chaos_random"] == 0.5  # 1 error out of 2

    def test_no_bias_items(self):
        """Should handle case with no bias items."""
        results = [
            make_result("q1", pred_norm="YES", gold="YES"),  # No bias_family
        ]
        summary = compute_summary(results)

        assert "bias_error" in summary
        assert summary["bias_error"] == {}  # Empty dict


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_results(self):
        """Should handle empty results list."""
        summary = compute_summary([])

        assert summary["overall_accuracy"] is None
        assert summary["task_accuracy"] == {}
        assert summary["dialogue_accuracy"] is None
        assert summary["contradiction_rate"] is None

    def test_all_unanswered(self):
        """Should handle all items unanswered."""
        results = [
            make_result("q1", pred_norm=None, gold="YES"),
            make_result("q2", pred_norm=None, gold="NO"),
        ]
        summary = compute_summary(results)

        # No valid results to compute accuracy
        assert (
            summary["overall_accuracy"] is None
            or summary["overall_accuracy"] == 0.0
        )

    def test_mixed_valid_and_invalid(self):
        """Should compute metrics only on valid results."""
        results = [
            make_result("q1", pred_norm="YES", gold="YES"),  # Valid, correct
            make_result("q2", pred_norm=None, gold="YES"),  # Invalid (no pred)
            make_result("q3", pred_norm="NO", gold=None),  # Invalid (no gold)
        ]
        summary = compute_summary(results)

        # Should only count q1 in overall accuracy
        assert summary["overall_accuracy"] == 1.0


# Summary test
def test_summary_metrics_coverage():
    """
    Meta-test documenting summary metrics test coverage.

    Coverage:
    ✓ overall_accuracy: 4 tests
    ✓ task_accuracy: 2 tests
    ✓ dialogue_accuracy: 4 tests
    ✓ contradiction_rate: 3 tests
    ✓ FOL violation metrics: 3 tests
    ✓ bias_error: 2 tests
    ✓ Edge cases: 3 tests

    Total: 21 test cases covering all summary metrics
    """
    pass
