"""Tests for bifurcation data and regime transition task in ChaosBench-Logic v2.

Covers BIFURCATION_DATA structure, regime lookups, is_chaotic_regime,
edge cases for get_regime_at_param, and question generation via
generate_regime_questions and RegimeTransitionTask.
"""

import pytest

from chaosbench.data.bifurcations import (
    BIFURCATION_DATA,
    BifurcationInfo,
    Transition,
    get_all_transitions,
    get_regime_at_param,
    is_chaotic_regime,
)
from chaosbench.data.schemas import Question
from chaosbench.tasks.regime_transition import (
    RegimeTransitionTask,
    generate_regime_questions,
)


class TestBifurcationData:
    """Structure and content of BIFURCATION_DATA."""

    def test_all_three_systems_present(self):
        """BIFURCATION_DATA contains logistic, lorenz63, and henon."""
        assert "logistic" in BIFURCATION_DATA
        assert "lorenz63" in BIFURCATION_DATA
        assert "henon" in BIFURCATION_DATA

    def test_exactly_seven_systems(self):
        """BIFURCATION_DATA has exactly 7 entries."""
        assert len(BIFURCATION_DATA) == 7

    def test_logistic_parameter_name(self):
        """Logistic map bifurcation parameter is r."""
        assert BIFURCATION_DATA["logistic"].parameter_name == "r"

    def test_lorenz63_parameter_name(self):
        """Lorenz63 bifurcation parameter is rho."""
        assert BIFURCATION_DATA["lorenz63"].parameter_name == "rho"

    def test_henon_parameter_name(self):
        """Henon map bifurcation parameter is a."""
        assert BIFURCATION_DATA["henon"].parameter_name == "a"

    def test_transitions_ordered_logistic(self):
        """Logistic transitions are ordered by param_value."""
        transitions = BIFURCATION_DATA["logistic"].transitions
        values = [t.param_value for t in transitions]
        assert values == sorted(values)

    def test_transitions_ordered_lorenz63(self):
        """Lorenz63 transitions are ordered by param_value."""
        transitions = BIFURCATION_DATA["lorenz63"].transitions
        values = [t.param_value for t in transitions]
        assert values == sorted(values)

    def test_transitions_ordered_henon(self):
        """Henon transitions are ordered by param_value."""
        transitions = BIFURCATION_DATA["henon"].transitions
        values = [t.param_value for t in transitions]
        assert values == sorted(values)

    def test_logistic_r4_full_chaos(self):
        """Logistic map at r=4 is in the full_chaos regime."""
        regime = get_regime_at_param("logistic", "r", 4.0)
        assert regime == "full_chaos"

    def test_logistic_r2_8_fixed_point(self):
        """Logistic map at r=2.8 is in the fixed_point regime."""
        regime = get_regime_at_param("logistic", "r", 2.8)
        assert regime == "fixed_point"

    def test_lorenz_rho28_classical_chaos(self):
        """Lorenz system at rho=28 is in classical_chaos regime."""
        regime = get_regime_at_param("lorenz63", "rho", 28.0)
        assert regime == "classical_chaos"

    def test_henon_a1_4_strange_attractor(self):
        """Henon map at a=1.4 is in strange_attractor regime."""
        regime = get_regime_at_param("henon", "a", 1.4)
        assert regime == "strange_attractor"


class TestIsChaoticRegime:
    """Detection of chaotic regime labels."""

    def test_chaos_onset_is_chaotic(self):
        """chaos_onset contains 'chaos' and is chaotic."""
        assert is_chaotic_regime("chaos_onset") is True

    def test_full_chaos_is_chaotic(self):
        """full_chaos contains 'chaos' and is chaotic."""
        assert is_chaotic_regime("full_chaos") is True

    def test_sustained_chaos_is_chaotic(self):
        """sustained_chaos contains 'chaos' and is chaotic."""
        assert is_chaotic_regime("sustained_chaos") is True

    def test_classical_chaos_is_chaotic(self):
        """classical_chaos contains 'chaos' and is chaotic."""
        assert is_chaotic_regime("classical_chaos") is True

    def test_fixed_point_not_chaotic(self):
        """fixed_point does not indicate chaos."""
        assert is_chaotic_regime("fixed_point") is False

    def test_period_2_not_chaotic(self):
        """period_2 does not indicate chaos."""
        assert is_chaotic_regime("period_2") is False

    def test_unknown_not_chaotic(self):
        """unknown regime label does not indicate chaos."""
        assert is_chaotic_regime("unknown") is False

    def test_strange_attractor_not_chaotic_by_name(self):
        """strange_attractor does not contain 'chaos' substring."""
        assert is_chaotic_regime("strange_attractor") is False


class TestGetRegimeEdgeCases:
    """Edge cases for get_regime_at_param."""

    def test_below_all_transitions(self):
        """Parameter below all transition points returns 'unknown'."""
        regime = get_regime_at_param("logistic", "r", -1.0)
        assert regime == "unknown"

    def test_at_exact_transition_point(self):
        """At exact transition boundary, returns that transition's regime."""
        regime = get_regime_at_param("logistic", "r", 3.0)
        assert regime == "period_2"

    def test_unknown_system_raises_keyerror(self):
        """Unknown system_id raises KeyError."""
        with pytest.raises(KeyError, match="Unknown system"):
            get_regime_at_param("nonexistent_system", "x", 1.0)

    def test_wrong_param_name_raises_valueerror(self):
        """Wrong parameter name raises ValueError."""
        with pytest.raises(ValueError, match="does not match"):
            get_regime_at_param("logistic", "sigma", 3.5)

    def test_get_all_transitions_returns_list(self):
        """get_all_transitions returns a list of Transition objects."""
        transitions = get_all_transitions("lorenz63")
        assert isinstance(transitions, list)
        assert all(isinstance(t, Transition) for t in transitions)

    def test_get_all_transitions_unknown_raises(self):
        """get_all_transitions raises KeyError for unknown system."""
        with pytest.raises(KeyError, match="Unknown system"):
            get_all_transitions("fake_system")


class TestQuestionGeneration:
    """Regime transition question generation."""

    def test_produces_questions(self):
        """generate_regime_questions produces a non-empty list."""
        questions = generate_regime_questions(BIFURCATION_DATA, seed=42)
        assert len(questions) > 0

    def test_questions_are_valid(self):
        """All generated questions pass validation."""
        questions = generate_regime_questions(BIFURCATION_DATA, seed=42)
        for q in questions:
            assert isinstance(q, Question)
            errors = q.validate()
            assert errors == [], f"Question {q.item_id} has errors: {errors}"

    def test_deterministic_with_seed(self):
        """Same seed produces identical question lists."""
        q1 = generate_regime_questions(BIFURCATION_DATA, seed=42)
        q2 = generate_regime_questions(BIFURCATION_DATA, seed=42)
        assert len(q1) == len(q2)
        for a, b in zip(q1, q2):
            assert a.item_id == b.item_id
            assert a.ground_truth == b.ground_truth
            assert a.question_text == b.question_text

    def test_covers_all_three_systems(self):
        """Questions span all three systems in BIFURCATION_DATA."""
        questions = generate_regime_questions(BIFURCATION_DATA, seed=42)
        covered = set()
        for q in questions:
            covered.add(q.system_id)
        assert "logistic" in covered
        assert "lorenz63" in covered
        assert "henon" in covered

    def test_ground_truth_matches_regime(self):
        """For is_chaotic questions, ground truth matches is_chaotic_regime."""
        questions = generate_regime_questions(BIFURCATION_DATA, seed=42)
        for q in questions:
            qtype = q.metadata.get("question_type", "")
            if qtype == "is_chaotic":
                regime = q.metadata["regime"]
                expected = "YES" if is_chaotic_regime(regime) else "NO"
                assert q.ground_truth == expected, (
                    f"{q.item_id}: regime={regime}, "
                    f"expected={expected}, got={q.ground_truth}"
                )

    def test_task_score_perfect(self):
        """RegimeTransitionTask scoring with all correct predictions."""
        task = RegimeTransitionTask(seed=42)
        items = task.generate_items()
        predictions = {q.item_id: q.ground_truth for q in items}
        result = task.score(predictions)
        assert result["accuracy"] == 1.0
        assert result["correct"] == result["total"]

    def test_task_score_empty_predictions(self):
        """RegimeTransitionTask scoring with no predictions returns zero."""
        task = RegimeTransitionTask(seed=42)
        result = task.score({})
        assert result["total"] == 0
        assert result["accuracy"] == 0.0

    def test_task_family_label(self):
        """All generated questions have task_family 'regime_transition'."""
        questions = generate_regime_questions(BIFURCATION_DATA, seed=42)
        for q in questions:
            assert q.task_family == "regime_transition"
