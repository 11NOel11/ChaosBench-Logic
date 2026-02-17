"""Tests for perturbation robustness task in ChaosBench-Logic v2."""

import pytest

from chaosbench.data.schemas import Question
from chaosbench.tasks.perturbation_robustness import generate_perturbation_questions


class TestPerturbationRobustness:
    """Perturbation robustness question generation."""

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

    def test_generate_perturbation_questions(self, sample_systems):
        """Basic generation works."""
        questions = generate_perturbation_questions(sample_systems, seed=42)

        assert isinstance(questions, list), "Should return a list"
        assert len(questions) > 0, "Should generate at least one question"

    def test_perturbation_fields(self, sample_systems):
        """Question fields correct."""
        questions = generate_perturbation_questions(sample_systems, seed=42)

        for q in questions:
            assert isinstance(q, Question), "Each item should be a Question object"
            assert q.item_id is not None, "item_id should not be None"
            assert q.question_text is not None, "question_text should not be None"
            assert q.system_id is not None, "system_id should not be None"
            assert q.ground_truth is not None, "ground_truth should not be None"
            assert q.task_family == "perturbation", "task_family should be 'perturbation'"

    def test_perturbation_types_in_metadata(self, sample_systems):
        """metadata has perturbation_type."""
        questions = generate_perturbation_questions(sample_systems, seed=42)

        for q in questions:
            assert "perturbation_type" in q.metadata, (
                "metadata should have perturbation_type field"
            )

            assert q.metadata["perturbation_type"] in [
                "paraphrase", "negation", "entity_swap", "distractor"
            ], "perturbation_type should be one of the four supported types"

    def test_perturbation_ground_truth(self, sample_systems):
        """YES or NO only."""
        questions = generate_perturbation_questions(sample_systems, seed=42)

        for q in questions:
            assert q.ground_truth in ("YES", "NO"), (
                f"ground_truth should be YES or NO, got {q.ground_truth}"
            )

    def test_perturbation_target_count(self, sample_systems):
        """Respects target_count."""
        questions = generate_perturbation_questions(
            sample_systems, seed=42, target_count=10
        )

        assert len(questions) <= 10, (
            f"Should return at most 10 questions, got {len(questions)}"
        )

    def test_perturbation_deterministic(self, sample_systems):
        """Same seed = same output."""
        q1 = generate_perturbation_questions(sample_systems, seed=42)
        q2 = generate_perturbation_questions(sample_systems, seed=42)

        assert len(q1) == len(q2), "Same seed should produce same number of questions"

        for a, b in zip(q1, q2):
            assert a.item_id == b.item_id, "item_id should match"
            assert a.ground_truth == b.ground_truth, "ground_truth should match"

    def test_perturbation_type_filter(self, sample_systems):
        """Passing specific perturbation_types filters correctly."""
        questions = generate_perturbation_questions(
            sample_systems,
            seed=42,
            perturbation_types=["paraphrase", "negation"]
        )

        for q in questions:
            assert q.metadata["perturbation_type"] in ["paraphrase", "negation"], (
                "Should only have paraphrase and negation perturbations"
            )
