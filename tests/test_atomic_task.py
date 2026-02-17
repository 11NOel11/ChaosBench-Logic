"""Tests for atomic task question generation in ChaosBench-Logic v2."""

import pytest

from chaosbench.data.schemas import Question
from chaosbench.tasks.atomic import generate_atomic_questions


class TestAtomicTaskGeneration:
    """Atomic task question generation."""

    @pytest.fixture
    def sample_systems(self):
        """Minimal systems dict for testing."""
        return {
            "lorenz63": {
                "name": "Lorenz-63 system",
                "truth_assignment": {
                    "Chaotic": True,
                    "Deterministic": True,
                    "PosLyap": True,
                    "Sensitive": True,
                    "StrangeAttr": True,
                    "PointUnpredictable": True,
                    "StatPredictable": True,
                    "QuasiPeriodic": False,
                    "Random": False,
                    "FixedPointAttr": False,
                    "Periodic": False,
                },
            },
            "shm": {
                "name": "Simple harmonic oscillator",
                "truth_assignment": {
                    "Chaotic": False,
                    "Deterministic": True,
                    "PosLyap": False,
                    "Sensitive": False,
                    "StrangeAttr": False,
                    "PointUnpredictable": False,
                    "StatPredictable": False,
                    "QuasiPeriodic": False,
                    "Random": False,
                    "FixedPointAttr": False,
                    "Periodic": True,
                },
            },
            "rossler": {
                "name": "Rossler system",
                "truth_assignment": {
                    "Chaotic": True,
                    "Deterministic": True,
                    "PosLyap": True,
                    "Sensitive": True,
                    "StrangeAttr": True,
                    "PointUnpredictable": True,
                    "StatPredictable": True,
                    "QuasiPeriodic": False,
                    "Random": False,
                    "FixedPointAttr": False,
                    "Periodic": False,
                },
            },
        }

    def test_generate_atomic_questions(self, sample_systems):
        """Call generate_atomic_questions with 3-system dict, verify returns list of Question."""
        questions = generate_atomic_questions(sample_systems, seed=42)

        assert isinstance(questions, list), "Should return a list"
        assert len(questions) > 0, "Should generate at least one question"

        for q in questions:
            assert isinstance(q, Question), "Each item should be a Question object"

    def test_atomic_question_fields(self, sample_systems):
        """Verify each Question has item_id, question_text, system_id, ground_truth."""
        questions = generate_atomic_questions(sample_systems, seed=42)

        for q in questions:
            assert q.item_id is not None, "item_id should not be None"
            assert q.question_text is not None, "question_text should not be None"
            assert q.system_id is not None, "system_id should not be None"
            assert q.ground_truth is not None, "ground_truth should not be None"

    def test_atomic_ground_truth_values(self, sample_systems):
        """Verify ground_truth is YES or NO."""
        questions = generate_atomic_questions(sample_systems, seed=42)

        for q in questions:
            assert q.ground_truth in ("YES", "NO"), (
                f"ground_truth should be YES or NO, got {q.ground_truth}"
            )

    def test_atomic_balanced(self, sample_systems):
        """Verify roughly balanced YES/NO (within 60/40)."""
        questions = generate_atomic_questions(sample_systems, seed=42)

        yes_count = sum(1 for q in questions if q.ground_truth == "YES")
        no_count = sum(1 for q in questions if q.ground_truth == "NO")
        total = len(questions)

        yes_ratio = yes_count / total if total > 0 else 0

        # Allow 60/40 balance (0.4 to 0.6)
        assert 0.4 <= yes_ratio <= 0.6, (
            f"YES/NO balance should be within 60/40, got {yes_ratio:.2f}"
        )

    def test_atomic_target_count(self, sample_systems):
        """Call with target_count=10, verify <= 10 returned."""
        questions = generate_atomic_questions(sample_systems, seed=42, target_count=10)

        assert len(questions) <= 10, f"Should return at most 10 questions, got {len(questions)}"

    def test_atomic_deterministic(self, sample_systems):
        """Same seed produces identical output."""
        q1 = generate_atomic_questions(sample_systems, seed=42)
        q2 = generate_atomic_questions(sample_systems, seed=42)

        assert len(q1) == len(q2), "Same seed should produce same number of questions"

        for a, b in zip(q1, q2):
            assert a.item_id == b.item_id, "item_id should match"
            assert a.question_text == b.question_text, "question_text should match"
            assert a.ground_truth == b.ground_truth, "ground_truth should match"
