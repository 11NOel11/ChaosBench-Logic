"""Tests for FOL inference task in ChaosBench-Logic v2."""

import pytest

from chaosbench.data.schemas import Question
from chaosbench.tasks.fol_inference import (
    FOLInferenceTask,
    generate_fol_questions,
)


class TestFOLQuestionGeneration:
    """FOL inference question generation."""

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

    def test_generates_questions(self, sample_systems):
        """generate_fol_questions produces a non-empty list."""
        questions = generate_fol_questions(sample_systems, seed=42)
        assert len(questions) > 0

    def test_questions_are_valid(self, sample_systems):
        """All generated questions pass validation."""
        questions = generate_fol_questions(sample_systems, seed=42)
        for q in questions:
            assert isinstance(q, Question)
            errors = q.validate()
            assert errors == [], f"Question {q.item_id} has errors: {errors}"

    def test_ground_truth_is_yes_or_no(self, sample_systems):
        """Ground truth labels are always YES or NO."""
        questions = generate_fol_questions(sample_systems, seed=42)
        for q in questions:
            assert q.ground_truth in ("YES", "NO"), (
                f"{q.item_id}: got {q.ground_truth}"
            )

    def test_deterministic_with_seed(self, sample_systems):
        """Same seed produces identical question lists."""
        q1 = generate_fol_questions(sample_systems, seed=42)
        q2 = generate_fol_questions(sample_systems, seed=42)
        assert len(q1) == len(q2)
        for a, b in zip(q1, q2):
            assert a.item_id == b.item_id
            assert a.ground_truth == b.ground_truth

    def test_item_ids_are_unique(self, sample_systems):
        """All FOL question item IDs are unique."""
        questions = generate_fol_questions(sample_systems, seed=42)
        item_ids = [q.item_id for q in questions]
        assert len(item_ids) == len(set(item_ids)), "Duplicate item IDs found"

    def test_has_multiple_question_types(self, sample_systems):
        """Questions span multiple FOL question types."""
        questions = generate_fol_questions(sample_systems, seed=42)
        types = {q.metadata.get("question_type") for q in questions}
        assert len(types) >= 3

    def test_implication_ground_truth_correct(self, sample_systems):
        """Implication questions about chaotic systems answer YES."""
        questions = generate_fol_questions(sample_systems, seed=42)
        impl_qs = [q for q in questions if q.metadata.get("question_type") == "implication"]
        for q in impl_qs:
            assert q.ground_truth == "YES"

    def test_exclusion_ground_truth_correct(self, sample_systems):
        """Exclusion questions always answer NO."""
        questions = generate_fol_questions(sample_systems, seed=42)
        excl_qs = [q for q in questions if q.metadata.get("question_type") == "exclusion"]
        for q in excl_qs:
            assert q.ground_truth == "NO"

    def test_contrapositive_ground_truth_correct(self, sample_systems):
        """Contrapositive questions always answer NO."""
        questions = generate_fol_questions(sample_systems, seed=42)
        contra_qs = [q for q in questions if q.metadata.get("question_type") == "contrapositive"]
        for q in contra_qs:
            assert q.ground_truth == "NO"

    def test_task_family_is_fol_inference(self, sample_systems):
        """All questions have task_family='fol_inference'."""
        questions = generate_fol_questions(sample_systems, seed=42)
        for q in questions:
            assert q.task_family == "fol_inference"

    def test_task_score_perfect(self, sample_systems):
        """Scoring with all correct predictions returns accuracy 1.0."""
        task = FOLInferenceTask(systems=sample_systems, seed=42)
        items = task.generate_items()
        predictions = {q.item_id: q.ground_truth for q in items}
        result = task.score(predictions)
        assert result["accuracy"] == 1.0
        assert result["correct"] == result["total"]


class TestFOLEdgeCases:
    """Edge case handling for FOL inference task."""

    def test_empty_systems_dict(self):
        """Handle empty systems dict gracefully."""
        questions = generate_fol_questions({}, seed=42)
        # Should still generate generic FOL questions (exclusion, contrapositive, consistency)
        assert len(questions) > 0
        # All questions should be valid
        for q in questions:
            assert q.ground_truth in ("YES", "NO")

    def test_missing_truth_assignment(self):
        """Skip systems missing truth_assignment."""
        systems = {
            "test_system": {
                "name": "Test System",
                # No truth_assignment
            }
        }
        questions = generate_fol_questions(systems, seed=42)
        # Should generate generic questions but skip system-specific ones
        assert len(questions) > 0
        # No system-specific implication questions should be generated
        impl_qs = [q for q in questions if q.metadata.get("question_type") == "implication"]
        assert len(impl_qs) == 0

    def test_incomplete_truth_assignment(self):
        """Handle systems with incomplete truth assignments."""
        systems = {
            "partial_system": {
                "name": "Partial System",
                "truth_assignment": {
                    "Chaotic": True,
                    # Missing other predicates
                }
            }
        }
        questions = generate_fol_questions(systems, seed=42)
        # Should handle gracefully and generate what it can
        assert len(questions) > 0
        for q in questions:
            errors = q.validate()
            assert errors == []
