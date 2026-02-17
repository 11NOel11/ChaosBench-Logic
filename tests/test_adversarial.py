"""Tests for adversarial question generation and hard split identification."""

import pytest

from chaosbench.data.schemas import Question
from chaosbench.data.adversarial import (
    generate_misleading_premise,
    generate_near_miss,
    generate_predicate_confusion,
    generate_adversarial_set,
    BIFURCATION_DATA,
)
from chaosbench.tasks.hard_split import (
    identify_hard_items,
    create_hard_split,
    analyze_hard_characteristics,
)

SAMPLE_SYSTEMS = {
    "lorenz63": {
        "name": "Lorenz system",
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
        "name": "Simple harmonic oscillator",
        "truth": {
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


def make_question(
    item_id="q1",
    text="Is the system chaotic?",
    gt="YES",
    predicates=None,
    system_id="lorenz63",
):
    """Helper to create a test question."""
    if predicates is None:
        predicates = ["Chaotic"]
    return Question(
        item_id=item_id,
        question_text=text,
        system_id=system_id,
        task_family="atomic",
        ground_truth=gt,
        predicates=predicates,
    )


class TestMisleadingPremise:
    """Test misleading premise perturbation."""

    def test_generates_valid_question(self):
        """Should return a valid Question object."""
        q = make_question()
        truth = SAMPLE_SYSTEMS["lorenz63"]["truth"]
        modified = generate_misleading_premise(q, truth, seed=42)
        assert isinstance(modified, Question)
        errors = modified.validate()
        assert errors == []

    def test_preserves_ground_truth(self):
        """Ground truth should remain unchanged."""
        q = make_question(gt="NO")
        truth = SAMPLE_SYSTEMS["lorenz63"]["truth"]
        modified = generate_misleading_premise(q, truth, seed=42)
        assert modified.ground_truth == "NO"

    def test_added_premise_is_true(self):
        """Added premise should contain a true predicate from system truth."""
        q = make_question(predicates=["Chaotic"])
        truth = SAMPLE_SYSTEMS["lorenz63"]["truth"]
        modified = generate_misleading_premise(q, truth, seed=42)
        misleading_pred = modified.metadata.get("misleading_predicate")
        assert misleading_pred in truth
        assert truth[misleading_pred] is True

    def test_deterministic_with_seed(self):
        """Same seed should produce identical output."""
        q = make_question()
        truth = SAMPLE_SYSTEMS["lorenz63"]["truth"]
        m1 = generate_misleading_premise(q, truth, seed=42)
        m2 = generate_misleading_premise(q, truth, seed=42)
        assert m1.question_text == m2.question_text
        assert m1.metadata == m2.metadata

    def test_task_family_and_metadata(self):
        """Task family should be adversarial_misleading with correct metadata."""
        q = make_question()
        truth = SAMPLE_SYSTEMS["lorenz63"]["truth"]
        modified = generate_misleading_premise(q, truth, seed=42)
        assert modified.task_family == "adversarial_misleading"
        assert modified.metadata["adversarial_type"] == "misleading_premise"
        assert "misleading_predicate" in modified.metadata


class TestNearMiss:
    """Test near-miss parameter boundary questions."""

    def test_generates_valid_question(self):
        """Should return a valid Question object."""
        q = generate_near_miss(
            "lorenz63",
            "Lorenz system",
            SAMPLE_SYSTEMS["lorenz63"]["truth"],
            BIFURCATION_DATA,
            seed=42,
        )
        assert isinstance(q, Question)
        errors = q.validate()
        assert errors == []

    def test_task_family_is_adversarial_nearmiss(self):
        """Task family should be adversarial_nearmiss."""
        q = generate_near_miss(
            "lorenz63",
            "Lorenz system",
            SAMPLE_SYSTEMS["lorenz63"]["truth"],
            BIFURCATION_DATA,
            seed=42,
        )
        assert q.task_family == "adversarial_nearmiss"

    def test_ground_truth_is_yes_or_no(self):
        """Ground truth should be YES or NO."""
        q = generate_near_miss(
            "lorenz63",
            "Lorenz system",
            SAMPLE_SYSTEMS["lorenz63"]["truth"],
            BIFURCATION_DATA,
            seed=42,
        )
        assert q.ground_truth in ("YES", "NO")

    def test_parameter_value_near_boundary(self):
        """Generated parameter value should be near a boundary."""
        q = generate_near_miss(
            "lorenz63",
            "Lorenz system",
            SAMPLE_SYSTEMS["lorenz63"]["truth"],
            BIFURCATION_DATA,
            seed=42,
        )
        metadata = q.metadata
        if "boundary_param" in metadata and "query_param" in metadata:
            boundary = metadata["boundary_param"]
            query = metadata["query_param"]
            relative_diff = abs(query - boundary) / max(abs(boundary), 0.01)
            assert relative_diff < 0.05

    def test_deterministic_with_seed(self):
        """Same seed should produce identical output."""
        q1 = generate_near_miss(
            "lorenz63",
            "Lorenz system",
            SAMPLE_SYSTEMS["lorenz63"]["truth"],
            BIFURCATION_DATA,
            seed=42,
        )
        q2 = generate_near_miss(
            "lorenz63",
            "Lorenz system",
            SAMPLE_SYSTEMS["lorenz63"]["truth"],
            BIFURCATION_DATA,
            seed=42,
        )
        assert q1.question_text == q2.question_text
        assert q1.ground_truth == q2.ground_truth
        assert q1.metadata == q2.metadata


class TestPredicateConfusion:
    """Test predicate confusion perturbation."""

    def test_swaps_to_confusable_predicate(self):
        """Should swap to a confusable predicate from CONFUSABLE_PAIRS."""
        q = make_question(predicates=["Chaotic"])
        modified = generate_predicate_confusion(q, seed=42)
        if modified.item_id != q.item_id:
            swapped = modified.metadata.get("swapped_predicate")
            assert swapped in ("Random", "StrangeAttr")

    def test_task_family_is_adversarial_confusion(self):
        """Task family should be adversarial_confusion."""
        q = make_question(predicates=["Chaotic"])
        modified = generate_predicate_confusion(q, seed=42)
        if modified.item_id != q.item_id:
            assert modified.task_family == "adversarial_confusion"

    def test_ground_truth_is_unknown(self):
        """Ground truth should be UNKNOWN after confusion."""
        q = make_question(predicates=["Chaotic"])
        modified = generate_predicate_confusion(q, seed=42)
        if modified.item_id != q.item_id:
            assert modified.ground_truth == "UNKNOWN"

    def test_returns_deepcopy_when_no_confusable(self):
        """When no confusable pair exists, should return deepcopy."""
        q = make_question(predicates=["PosLyap"])
        modified = generate_predicate_confusion(q, seed=42)
        assert modified.item_id == q.item_id
        assert modified.question_text == q.question_text


class TestGenerateAdversarialSet:
    """Test adversarial question set generation."""

    def test_returns_three_times_n_per_type(self):
        """Should return 3 * n_per_type questions."""
        n = 5
        questions = generate_adversarial_set(SAMPLE_SYSTEMS, n_per_type=n, seed=42)
        assert len(questions) == 3 * n

    def test_mix_of_all_three_types(self):
        """Should contain all three adversarial types."""
        questions = generate_adversarial_set(SAMPLE_SYSTEMS, n_per_type=3, seed=42)
        families = {q.task_family for q in questions}
        assert "adversarial_misleading" in families
        assert "adversarial_nearmiss" in families
        assert "adversarial_confusion" in families

    def test_deterministic_with_seed(self):
        """Same seed should produce identical output."""
        q1 = generate_adversarial_set(SAMPLE_SYSTEMS, n_per_type=3, seed=42)
        q2 = generate_adversarial_set(SAMPLE_SYSTEMS, n_per_type=3, seed=42)
        assert len(q1) == len(q2)
        for qa, qb in zip(q1, q2):
            assert qa.item_id == qb.item_id
            assert qa.question_text == qb.question_text
            assert qa.ground_truth == qb.ground_truth

    def test_all_questions_have_valid_item_ids(self):
        """All questions should have non-empty item_ids."""
        questions = generate_adversarial_set(SAMPLE_SYSTEMS, n_per_type=3, seed=42)
        for q in questions:
            assert q.item_id
            assert isinstance(q.item_id, str)
            assert len(q.item_id) > 0


class TestHardSplit:
    """Test hard split identification and analysis."""

    def test_identify_hard_items_returns_empty_for_nonexistent_dir(self):
        """Should return empty list for nonexistent directory."""
        hard = identify_hard_items(results_dir="/nonexistent/path/xyz", threshold=0.6)
        assert hard == []

    def test_identify_hard_items_with_published_results(self):
        """Should identify hard task families from published results."""
        hard = identify_hard_items(
            results_dir="/Users/noel.thomas/chaos-logic-bench/published_results",
            threshold=0.6,
        )
        assert isinstance(hard, list)
        assert all(isinstance(f, str) for f in hard)
        assert hard == sorted(hard)

    def test_create_hard_split_filters_by_task_family(self):
        """Should filter questions by hard task families."""
        questions = [
            make_question(item_id="q1", text="test 1"),
            make_question(item_id="q2", text="test 2"),
            make_question(item_id="q3", text="test 3"),
        ]
        questions[0].task_family = "atomic"
        questions[1].task_family = "trap"
        questions[2].task_family = "hard"

        hard_families = ["trap", "hard"]
        hard = create_hard_split(questions, hard_families)
        assert len(hard) == 2
        assert hard[0].task_family in hard_families
        assert hard[1].task_family in hard_families

    def test_analyze_hard_characteristics_returns_expected_structure(self):
        """Should return dict with hard and all keys containing stats."""
        all_items = [
            make_question(item_id=f"q{i}", predicates=["Chaotic"])
            for i in range(10)
        ]
        hard_items = all_items[:3]

        analysis = analyze_hard_characteristics(hard_items, all_items)
        assert "hard" in analysis
        assert "all" in analysis
        assert "task_family_dist" in analysis["hard"]
        assert "predicate_dist" in analysis["hard"]
        assert "avg_question_length" in analysis["hard"]
        assert "count" in analysis["hard"]
        assert analysis["hard"]["count"] == 3
        assert analysis["all"]["count"] == 10

    def test_analyze_empty_items_returns_correct_defaults(self):
        """Empty items should return zero counts and empty dists."""
        analysis = analyze_hard_characteristics([], [])
        assert analysis["hard"]["count"] == 0
        assert analysis["hard"]["task_family_dist"] == {}
        assert analysis["hard"]["predicate_dist"] == {}
        assert analysis["hard"]["avg_question_length"] == 0.0

    def test_hard_split_preserves_question_objects(self):
        """Hard split should preserve original question objects unchanged."""
        q1 = make_question(item_id="q1")
        q1.task_family = "trap"
        questions = [q1]
        hard = create_hard_split(questions, ["trap"])
        assert len(hard) == 1
        assert hard[0].item_id == "q1"
        assert hard[0].question_text == q1.question_text
        assert hard[0].ground_truth == q1.ground_truth

    def test_identify_hard_items_respects_threshold(self):
        """Should only return families below threshold."""
        hard_low = identify_hard_items(
            results_dir="/Users/noel.thomas/chaos-logic-bench/published_results",
            threshold=0.3,
        )
        hard_high = identify_hard_items(
            results_dir="/Users/noel.thomas/chaos-logic-bench/published_results",
            threshold=0.9,
        )
        assert len(hard_low) <= len(hard_high)
