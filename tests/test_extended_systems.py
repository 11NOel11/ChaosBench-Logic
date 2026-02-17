"""Tests for extended systems task in ChaosBench-Logic v2."""

import pytest

from chaosbench.data.schemas import Question
from chaosbench.tasks.extended_systems import (
    ExtendedSystemsTask,
    generate_extended_system_questions,
    TARGET_SYSTEMS,
)


class TestExtendedSystemsGeneration:
    """Extended systems question generation."""

    @pytest.fixture
    def sample_systems(self):
        """Systems dict covering a few target systems."""
        return {
            "sine_gordon": {
                "name": "Sine-Gordon equation",
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
            "ikeda_map": {
                "name": "Ikeda map",
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
            "brusselator": {
                "name": "Brusselator",
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
        """generate_extended_system_questions produces a non-empty list."""
        questions = generate_extended_system_questions(sample_systems, seed=42)
        assert len(questions) > 0

    def test_questions_are_valid(self, sample_systems):
        """All generated questions pass validation."""
        questions = generate_extended_system_questions(sample_systems, seed=42)
        for q in questions:
            assert isinstance(q, Question)
            errors = q.validate()
            assert errors == [], f"Question {q.item_id} has errors: {errors}"

    def test_ground_truth_is_yes_or_no(self, sample_systems):
        """Ground truth labels are always YES or NO."""
        questions = generate_extended_system_questions(sample_systems, seed=42)
        for q in questions:
            assert q.ground_truth in ("YES", "NO")

    def test_deterministic_with_seed(self, sample_systems):
        """Same seed produces identical questions."""
        q1 = generate_extended_system_questions(sample_systems, seed=42)
        q2 = generate_extended_system_questions(sample_systems, seed=42)
        assert len(q1) == len(q2)
        for a, b in zip(q1, q2):
            assert a.item_id == b.item_id
            assert a.ground_truth == b.ground_truth

    def test_three_questions_per_system(self, sample_systems):
        """Generates ~3 questions per system (expanded from 2)."""
        questions = generate_extended_system_questions(sample_systems, seed=42)
        systems_covered = {q.system_id for q in questions}
        assert len(questions) == len(systems_covered) * 3

    def test_item_ids_are_unique(self, sample_systems):
        """All item IDs are unique within the task."""
        questions = generate_extended_system_questions(sample_systems, seed=42)
        item_ids = [q.item_id for q in questions]
        assert len(item_ids) == len(set(item_ids)), "Duplicate item IDs found"

    def test_task_family_is_extended_systems(self, sample_systems):
        """All questions have correct task_family."""
        questions = generate_extended_system_questions(sample_systems, seed=42)
        for q in questions:
            assert q.task_family == "extended_systems"

    def test_task_score_perfect(self, sample_systems):
        """Scoring with all correct predictions returns accuracy 1.0."""
        task = ExtendedSystemsTask(systems=sample_systems, seed=42)
        items = task.generate_items()
        predictions = {q.item_id: q.ground_truth for q in items}
        result = task.score(predictions)
        assert result["accuracy"] == 1.0

    def test_target_systems_list_has_entries(self):
        """TARGET_SYSTEMS is non-empty."""
        assert len(TARGET_SYSTEMS) >= 10


class TestExtendedSystemsEdgeCases:
    """Edge case handling for extended systems task."""

    def test_empty_systems_dict(self):
        """Handle empty systems dict gracefully."""
        questions = generate_extended_system_questions({}, seed=42)
        assert questions == []

    def test_missing_truth_assignment(self):
        """Skip systems missing truth_assignment."""
        systems = {
            "test_system": {
                "name": "Test System",
                # No truth_assignment
            }
        }
        questions = generate_extended_system_questions(systems, seed=42)
        # Should skip system with missing truth assignment
        assert questions == []

    def test_system_not_in_target_list(self):
        """Only generate questions for systems in TARGET_SYSTEMS."""
        systems = {
            "unknown_system": {
                "name": "Unknown System",
                "truth_assignment": {"Chaotic": True, "Deterministic": True},
            }
        }
        questions = generate_extended_system_questions(systems, seed=42)
        # Should skip systems not in TARGET_SYSTEMS
        assert questions == []
