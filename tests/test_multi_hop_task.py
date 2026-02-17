"""Tests for multi-hop task question generation in ChaosBench-Logic v2."""

import pytest

from chaosbench.data.schemas import Question
from chaosbench.tasks.multi_hop import generate_multi_hop_questions


class TestMultiHopTaskGeneration:
    """Multi-hop task question generation."""

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
        }

    def test_generate_multi_hop_questions(self, sample_systems):
        """Call generate_multi_hop_questions with system dict."""
        questions = generate_multi_hop_questions(sample_systems, seed=42)

        assert isinstance(questions, list), "Should return a list"
        assert len(questions) > 0, "Should generate at least one question"

    def test_multi_hop_fields(self, sample_systems):
        """Verify Question fields."""
        questions = generate_multi_hop_questions(sample_systems, seed=42)

        for q in questions:
            assert isinstance(q, Question), "Each item should be a Question object"
            assert q.item_id is not None, "item_id should not be None"
            assert q.question_text is not None, "question_text should not be None"
            assert q.system_id is not None, "system_id should not be None"
            assert q.ground_truth is not None, "ground_truth should not be None"
            assert q.task_family == "multi_hop", "task_family should be 'multi_hop'"

    def test_multi_hop_ground_truth_values(self, sample_systems):
        """YES or NO only."""
        questions = generate_multi_hop_questions(sample_systems, seed=42)

        for q in questions:
            assert q.ground_truth in ("YES", "NO"), (
                f"ground_truth should be YES or NO, got {q.ground_truth}"
            )

    def test_multi_hop_target_count(self, sample_systems):
        """target_count works."""
        questions = generate_multi_hop_questions(sample_systems, seed=42, target_count=10)

        assert len(questions) <= 10, f"Should return at most 10 questions, got {len(questions)}"

    def test_multi_hop_deterministic(self, sample_systems):
        """Same seed = same output."""
        q1 = generate_multi_hop_questions(sample_systems, seed=42)
        q2 = generate_multi_hop_questions(sample_systems, seed=42)

        assert len(q1) == len(q2), "Same seed should produce same number of questions"

        for a, b in zip(q1, q2):
            assert a.item_id == b.item_id, "item_id should match"
            assert a.question_text == b.question_text, "question_text should match"
            assert a.ground_truth == b.ground_truth, "ground_truth should match"

    def test_multi_hop_question_types(self, sample_systems):
        """metadata has question_type field."""
        questions = generate_multi_hop_questions(sample_systems, seed=42)

        for q in questions:
            assert "reasoning_type" in q.metadata or "hop_count" in q.metadata, (
                "metadata should have reasoning_type or hop_count field"
            )
