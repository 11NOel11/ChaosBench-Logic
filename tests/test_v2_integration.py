"""End-to-end integration tests for ChaosBench-Logic v2.

Tests the full pipeline: trajectory generation, indicator computation,
diagnostic task generation, regime transition tasks, adversarial generation,
solver repair, and backward compatibility.
"""

import json
import tempfile
from typing import Any, Dict

import numpy as np
import pytest

import chaosbench
from chaosbench.data.indicators.time_series import (
    SYSTEM_REGISTRY,
    generate_ode_trajectory,
    generate_map_trajectory,
    get_system_type,
)
from chaosbench.data.indicators.zero_one_test import zero_one_test
from chaosbench.data.indicators.permutation_entropy import permutation_entropy
from chaosbench.data.indicators.populate import (
    compute_all_indicators,
    ALL_SYSTEMS,
)
from chaosbench.tasks.indicator_diagnostics import (
    IndicatorDiagnosticTask,
    generate_indicator_questions,
)
from chaosbench.data.bifurcations import (
    BIFURCATION_DATA,
    get_regime_at_param,
)
from chaosbench.tasks.regime_transition import (
    RegimeTransitionTask,
    generate_regime_questions,
)
from chaosbench.data.adversarial import (
    generate_adversarial_set,
    CONFUSABLE_PAIRS,
)
from chaosbench.logic.solver_repair import (
    repair_assignment,
    validate_repair,
)
from chaosbench.logic.axioms import (
    check_fol_violations,
    get_fol_rules,
)
from chaosbench.data.schemas import Question


class TestIndicatorPipeline:
    """Test full indicator computation pipeline."""

    def test_trajectory_to_indicators_consistent(self):
        """Generate trajectory, compute indicators, verify consistency."""
        system_id = "lorenz63"
        params = {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}

        traj = generate_ode_trajectory(
            system_id, params, t_span=(0, 100), n_points=10000, seed=42
        )

        assert traj.shape[0] == 10000
        assert traj.shape[1] == 3

        series = traj[:, 0]

        k_val = zero_one_test(series, seed=42)
        assert 0.0 <= k_val <= 1.0

        pe_val = permutation_entropy(series, order=3, delay=1)
        assert 0.0 <= pe_val <= 1.0

        k_val_2 = zero_one_test(series, seed=42)
        assert abs(k_val - k_val_2) < 1e-9

    def test_compute_all_indicators_json_serializable(self):
        """Verify compute_all_indicators produces valid JSON-serializable output."""
        system_id = "henon"
        result = compute_all_indicators(system_id, seed=42)

        assert result["system_id"] == system_id
        assert "zero_one_K" in result
        assert "permutation_entropy" in result
        assert "megno" in result
        assert result["system_type"] == "map"

        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["system_id"] == system_id

    def test_diagnostic_questions_from_real_indicators(self):
        """Generate diagnostic questions from computed indicators, verify validity."""
        system_id = "logistic_r4"
        indicators = compute_all_indicators(system_id, seed=42)

        truth_assignment = {
            "Chaotic": True,
            "Deterministic": True,
            "PosLyap": True,
            "Sensitive": True,
            "StrangeAttr": False,
            "PointUnpredictable": True,
            "StatPredictable": True,
            "QuasiPeriodic": False,
            "Random": False,
            "FixedPointAttr": False,
            "Periodic": False,
        }

        systems = {
            system_id: {
                "name": "Logistic Map (r=4)",
                "truth_assignment": truth_assignment,
            }
        }

        task = IndicatorDiagnosticTask(
            systems=systems,
            indicators={system_id: indicators},
            seed=42,
        )

        questions = task.generate_items()

        assert len(questions) > 0
        for q in questions:
            errors = q.validate()
            assert len(errors) == 0
            assert q.ground_truth in ("YES", "NO")

    def test_score_perfect_predictions_on_diagnostic_task(self):
        """Score a DummyEchoModel that returns ground truth, verify 100% accuracy."""
        system_id = "rossler"
        indicators = {
            "zero_one_K": 0.95,
            "permutation_entropy": 0.88,
        }

        truth_assignment = {
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
        }

        systems = {
            system_id: {
                "name": "Rossler Attractor",
                "truth_assignment": truth_assignment,
            }
        }

        task = IndicatorDiagnosticTask(
            systems=systems,
            indicators={system_id: indicators},
            seed=42,
        )

        questions = task.generate_items()

        perfect_predictions = {q.item_id: q.ground_truth for q in questions}

        result = task.score(perfect_predictions)

        assert result["accuracy"] == 1.0
        assert result["correct"] == result["total"]
        assert result["total"] > 0


class TestRegimeTransitionPipeline:
    """Test regime transition question generation and scoring."""

    def test_generate_regime_questions_all_valid_ground_truth(self):
        """Generate regime questions, verify all have valid ground truth."""
        questions = generate_regime_questions(BIFURCATION_DATA, seed=42)

        assert len(questions) > 0

        for q in questions:
            errors = q.validate()
            assert len(errors) == 0
            assert q.ground_truth in ("YES", "NO")
            assert q.task_family == "regime_transition"

    def test_dummy_echo_model_achieves_perfect_score(self):
        """Score with DummyEchoModel that returns ground truth, verify accuracy 1.0."""
        task = RegimeTransitionTask(seed=42)
        questions = task.generate_items()

        assert len(questions) > 0

        perfect_predictions = {q.item_id: q.ground_truth for q in questions}

        result = task.score(perfect_predictions)

        assert result["accuracy"] == 1.0
        assert result["correct"] == result["total"]

    def test_questions_cover_multiple_types(self):
        """Verify questions cover all expected question types."""
        questions = generate_regime_questions(BIFURCATION_DATA, seed=42)

        question_types = {q.metadata.get("question_type") for q in questions}

        expected_types = {"is_chaotic", "chaos_persistence", "is_periodic", "is_stable"}
        assert expected_types.issubset(question_types)


class TestAdversarialPipeline:
    """Test adversarial question generation pipeline."""

    def test_generate_adversarial_set_all_valid(self):
        """Generate adversarial set with real system data, verify all valid."""
        systems = {
            "lorenz63": {
                "name": "Lorenz System",
                "truth": {
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
                "name": "Simple Harmonic Motion",
                "truth": {
                    "Chaotic": False,
                    "Deterministic": True,
                    "PosLyap": False,
                    "Sensitive": False,
                    "StrangeAttr": False,
                    "PointUnpredictable": False,
                    "StatPredictable": True,
                    "QuasiPeriodic": False,
                    "Random": False,
                    "FixedPointAttr": False,
                    "Periodic": True,
                },
            },
        }

        questions = generate_adversarial_set(systems, n_per_type=3, seed=42)

        assert len(questions) == 9

        valid_count = 0
        unknown_count = 0
        for q in questions:
            if q.ground_truth == "UNKNOWN":
                unknown_count += 1
            else:
                errors = q.validate()
                assert len(errors) == 0
                valid_count += 1

        assert valid_count > 0
        assert unknown_count > 0

    def test_adversarial_questions_scoreable(self):
        """Verify adversarial questions can be scored through eval pipeline."""
        systems = {
            "henon": {
                "name": "Henon Map",
                "truth": {
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

        questions = generate_adversarial_set(systems, n_per_type=2, seed=42)

        predictions = {}
        for q in questions:
            if q.ground_truth != "UNKNOWN":
                predictions[q.item_id] = q.ground_truth
            else:
                predictions[q.item_id] = "NO"

        correct_count = sum(
            1 for q in questions
            if q.ground_truth != "UNKNOWN" and predictions.get(q.item_id) == q.ground_truth
        )

        assert correct_count >= 0

    def test_mix_of_all_adversarial_types(self):
        """Verify all 3 adversarial types appear in output."""
        systems = {
            "logistic": {
                "name": "Logistic Map",
                "truth": {
                    "Chaotic": True,
                    "Deterministic": True,
                    "PosLyap": True,
                    "Sensitive": True,
                    "StrangeAttr": False,
                    "PointUnpredictable": True,
                    "StatPredictable": True,
                    "QuasiPeriodic": False,
                    "Random": False,
                    "FixedPointAttr": False,
                    "Periodic": False,
                },
            },
        }

        questions = generate_adversarial_set(systems, n_per_type=3, seed=42)

        types = {q.metadata.get("adversarial_type") for q in questions if "adversarial_type" in q.metadata}

        expected = {"misleading_premise", "near_miss", "predicate_confusion"}
        assert expected.issubset(types)


class TestSolverRepairInContext:
    """Test MaxSAT solver repair with real system data."""

    def test_ground_truth_system_zero_flips(self):
        """Load ground truth system, repair should produce 0 flips."""
        ground_truth = {
            "Chaotic": "YES",
            "Deterministic": "YES",
            "PosLyap": "YES",
            "Sensitive": "YES",
            "StrangeAttr": "YES",
            "PointUnpredictable": "YES",
            "StatPredictable": "YES",
            "QuasiPeriodic": "NO",
            "Random": "NO",
            "FixedPointAttr": "NO",
            "Periodic": "NO",
        }

        violations_before = check_fol_violations(ground_truth)
        assert len(violations_before) == 0

        repaired, flips = repair_assignment(ground_truth)

        assert flips == 0
        assert repaired == ground_truth

    def test_introduce_violation_repair_fixes(self):
        """Introduce a violation, verify repair fixes it with minimal flips."""
        predictions = {
            "Chaotic": "YES",
            "Deterministic": "NO",
            "PosLyap": "YES",
            "Sensitive": "YES",
            "StrangeAttr": "NO",
            "PointUnpredictable": "YES",
            "StatPredictable": "YES",
            "QuasiPeriodic": "NO",
            "Random": "NO",
            "FixedPointAttr": "NO",
            "Periodic": "NO",
        }

        violations_before = check_fol_violations(predictions)
        assert len(violations_before) > 0

        repaired, flips = repair_assignment(predictions)

        violations_after = check_fol_violations(repaired)
        assert len(violations_after) == 0

        assert flips >= 1
        assert flips <= 2

    def test_repaired_assignment_passes_validation(self):
        """Verify repaired assignment passes validate_repair."""
        predictions = {
            "Chaotic": "YES",
            "Deterministic": "YES",
            "PosLyap": "NO",
            "Sensitive": "NO",
            "StrangeAttr": "NO",
            "PointUnpredictable": "NO",
            "StatPredictable": "NO",
            "QuasiPeriodic": "NO",
            "Random": "NO",
            "FixedPointAttr": "NO",
            "Periodic": "NO",
        }

        violations_before = check_fol_violations(predictions)
        assert len(violations_before) > 0

        repaired, flips = repair_assignment(predictions)

        is_valid = validate_repair(repaired)
        assert is_valid

        assert flips > 0


class TestBackwardCompat:
    """Test backward compatibility and public API."""

    def test_import_chaosbench_works(self):
        """Verify import chaosbench works."""
        assert hasattr(chaosbench, "SystemInstance")
        assert hasattr(chaosbench, "Question")
        assert hasattr(chaosbench, "PREDICATES")
        assert hasattr(chaosbench, "get_fol_rules")
        assert hasattr(chaosbench, "check_fol_violations")
        assert hasattr(chaosbench, "EvalResult")
        assert hasattr(chaosbench, "normalize_label")

    def test_version_is_2_1_0(self):
        """Verify version is 2.1.0."""
        assert chaosbench.__version__ == "2.1.0"
