"""Tests for cross-indicator task in ChaosBench-Logic v2."""

import pytest

from chaosbench.data.schemas import Question
from chaosbench.tasks.cross_indicator import (
    CrossIndicatorTask,
    generate_cross_indicator_questions,
)


class TestCrossIndicatorGeneration:
    """Cross-indicator question generation."""

    @pytest.fixture
    def sample_systems(self):
        """Minimal systems dict."""
        return {
            "lorenz63": {
                "name": "Lorenz-63 system",
                "truth_assignment": {"Chaotic": True},
            },
            "shm": {
                "name": "Simple harmonic oscillator",
                "truth_assignment": {"Chaotic": False},
            },
            "henon": {
                "name": "Henon map",
                "truth_assignment": {"Chaotic": True},
            },
        }

    @pytest.fixture
    def sample_indicators(self):
        """Minimal indicator values."""
        return {
            "lorenz63": {
                "system_id": "lorenz63",
                "zero_one_K": 0.95,
                "permutation_entropy": 0.88,
                "megno": 4.5,
            },
            "shm": {
                "system_id": "shm",
                "zero_one_K": 0.05,
                "permutation_entropy": 0.15,
                "megno": 2.0,
            },
            "henon": {
                "system_id": "henon",
                "zero_one_K": 0.92,
                "permutation_entropy": 0.85,
                "megno": 3.8,
            },
        }

    def test_generates_questions(self, sample_systems, sample_indicators):
        """generate_cross_indicator_questions produces a non-empty list."""
        questions = generate_cross_indicator_questions(
            sample_systems, sample_indicators, seed=42,
        )
        assert len(questions) > 0

    def test_questions_are_valid(self, sample_systems, sample_indicators):
        """All generated questions pass validation."""
        questions = generate_cross_indicator_questions(
            sample_systems, sample_indicators, seed=42,
        )
        for q in questions:
            assert isinstance(q, Question)
            errors = q.validate()
            assert errors == [], f"Question {q.item_id} has errors: {errors}"

    def test_ground_truth_is_yes_or_no(self, sample_systems, sample_indicators):
        """Ground truth labels are always YES or NO."""
        questions = generate_cross_indicator_questions(
            sample_systems, sample_indicators, seed=42,
        )
        for q in questions:
            assert q.ground_truth in ("YES", "NO")

    def test_item_ids_are_unique(self, sample_systems, sample_indicators):
        """All cross-indicator item IDs are unique."""
        questions = generate_cross_indicator_questions(
            sample_systems, sample_indicators, seed=42,
        )
        item_ids = [q.item_id for q in questions]
        assert len(item_ids) == len(set(item_ids)), "Duplicate item IDs found"

    def test_deterministic_with_seed(self, sample_systems, sample_indicators):
        """Same seed produces identical questions."""
        q1 = generate_cross_indicator_questions(
            sample_systems, sample_indicators, seed=42,
        )
        q2 = generate_cross_indicator_questions(
            sample_systems, sample_indicators, seed=42,
        )
        assert len(q1) == len(q2)
        for a, b in zip(q1, q2):
            assert a.item_id == b.item_id
            assert a.ground_truth == b.ground_truth

    def test_has_multiple_question_types(self, sample_systems, sample_indicators):
        """Questions span multiple cross-indicator types."""
        questions = generate_cross_indicator_questions(
            sample_systems, sample_indicators, seed=42,
        )
        types = {q.metadata.get("question_type") for q in questions}
        assert len(types) >= 2

    def test_task_family_is_cross_indicator(self, sample_systems, sample_indicators):
        """All questions have correct task_family."""
        questions = generate_cross_indicator_questions(
            sample_systems, sample_indicators, seed=42,
        )
        for q in questions:
            assert q.task_family == "cross_indicator"

    def test_task_score_perfect(self, sample_systems, sample_indicators):
        """Scoring with all correct predictions returns accuracy 1.0."""
        task = CrossIndicatorTask(
            systems=sample_systems,
            indicators=sample_indicators,
            seed=42,
        )
        items = task.generate_items()
        predictions = {q.item_id: q.ground_truth for q in items}
        result = task.score(predictions)
        assert result["accuracy"] == 1.0

    def test_consistency_question_logic(self, sample_systems, sample_indicators):
        """Consistency questions correctly identify aligned indicators."""
        questions = generate_cross_indicator_questions(
            sample_systems, sample_indicators, seed=42,
        )
        consist_qs = [
            q for q in questions
            if q.metadata.get("question_type") == "indicator_consistency"
        ]
        for q in consist_qs:
            k_val = q.metadata["indicator_value"]
            chaotic = q.metadata["system_chaotic"]
            k_suggests = k_val > 0.5
            expected = "YES" if (k_suggests == chaotic) else "NO"
            assert q.ground_truth == expected, (
                f"{q.item_id}: K={k_val}, chaotic={chaotic}, "
                f"expected={expected}, got={q.ground_truth}"
            )


class TestCrossIndicatorEdgeCases:
    """Edge case handling for cross-indicator task."""

    def test_empty_indicators_dict(self):
        """Handle empty indicators dict gracefully."""
        systems = {"test": {"name": "Test", "truth_assignment": {"Chaotic": True}}}
        questions = generate_cross_indicator_questions(systems, {}, seed=42)
        # Should return empty list when no indicators available
        assert questions == []

    def test_indicator_none_values(self):
        """Handle None indicator values gracefully."""
        systems = {"test": {"name": "Test", "truth_assignment": {"Chaotic": True}}}
        indicators = {
            "test": {
                "system_id": "test",
                "zero_one_K": None,  # None value
                "permutation_entropy": None,
                "megno": None,
            }
        }
        questions = generate_cross_indicator_questions(systems, indicators, seed=42)
        # Should skip None values and not crash
        # May have comparison questions if multiple systems
        for q in questions:
            assert q.ground_truth in ("YES", "NO")

    def test_mismatched_systems_and_indicators(self):
        """Handle case where systems and indicators don't match."""
        systems = {"sys1": {"name": "System 1", "truth_assignment": {"Chaotic": True}}}
        indicators = {"sys2": {"system_id": "sys2", "zero_one_K": 0.9}}
        questions = generate_cross_indicator_questions(systems, indicators, seed=42)
        # Should handle gracefully, no questions since no matching system
        # (or might have generic comparison questions)
        for q in questions:
            assert q.ground_truth in ("YES", "NO")
