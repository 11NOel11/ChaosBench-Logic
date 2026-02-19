"""Tests for chaosbench.eval.belief_dynamics -- belief divergence and instability."""

import pytest

from chaosbench.eval.belief_dynamics import (
    hamming_distance,
    belief_divergence_curve,
    instability_score,
    sensitivity_profile,
    belief_flip_rate,
    correlation_instability_accuracy,
)
from chaosbench.logic.extract import (
    extract_belief_vector,
    extract_belief_sequence,
    extract_predicate_from_question,
)


class TestHammingDistance:
    """Test Hamming distance between belief vectors."""

    def test_identical_beliefs(self):
        """Identical beliefs should have distance 0."""
        b = {"Chaotic": "YES", "Deterministic": "YES"}
        assert hamming_distance(b, b) == 0

    def test_completely_different(self):
        """All different beliefs should have maximum distance."""
        b1 = {"Chaotic": "YES", "Deterministic": "YES"}
        b2 = {"Chaotic": "NO", "Deterministic": "NO"}
        assert hamming_distance(b1, b2) == 2

    def test_partial_overlap(self):
        """Only shared keys should be compared."""
        b1 = {"Chaotic": "YES", "Random": "NO"}
        b2 = {"Chaotic": "NO", "Periodic": "YES"}
        assert hamming_distance(b1, b2) == 1

    def test_empty_beliefs(self):
        """Empty beliefs should have distance 0."""
        assert hamming_distance({}, {}) == 0

    def test_no_shared_keys(self):
        """Disjoint key sets should have distance 0."""
        b1 = {"Chaotic": "YES"}
        b2 = {"Random": "NO"}
        assert hamming_distance(b1, b2) == 0

    def test_unknown_counts_as_different(self):
        """UNKNOWN vs YES/NO should count as different."""
        b1 = {"Chaotic": "YES"}
        b2 = {"Chaotic": "UNKNOWN"}
        assert hamming_distance(b1, b2) == 1


class TestBeliefDivergenceCurve:
    """Test divergence curve computation."""

    def test_identical_sequences(self):
        """Identical sequences should have all-zero divergence."""
        b = [{"Chaotic": "YES"}, {"Chaotic": "YES"}]
        curve = belief_divergence_curve(b, b)
        assert curve == [0.0, 0.0]

    def test_diverging_sequences(self):
        """Diverging sequences should show increasing distance."""
        b_clean = [
            {"Chaotic": "YES", "Random": "NO"},
            {"Chaotic": "YES", "Random": "NO"},
        ]
        b_perturbed = [
            {"Chaotic": "YES", "Random": "NO"},
            {"Chaotic": "NO", "Random": "YES"},
        ]
        curve = belief_divergence_curve(b_clean, b_perturbed)
        assert curve[0] == 0.0
        assert curve[1] == 1.0

    def test_different_lengths(self):
        """Should handle sequences of different lengths."""
        b_clean = [{"Chaotic": "YES"}] * 3
        b_perturbed = [{"Chaotic": "NO"}] * 2
        curve = belief_divergence_curve(b_clean, b_perturbed)
        assert len(curve) == 2

    def test_empty_beliefs_in_turn(self):
        """Empty belief at a turn should produce 0.0 divergence."""
        b_clean = [{}]
        b_perturbed = [{}]
        curve = belief_divergence_curve(b_clean, b_perturbed)
        assert curve == [0.0]


class TestInstabilityScore:
    """Test instability score computation."""

    def test_no_curves(self):
        """No curves should give 0.0."""
        assert instability_score([]) == 0.0

    def test_zero_divergence(self):
        """All-zero curves should give 0.0 instability."""
        curves = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        assert instability_score(curves) == 0.0

    def test_full_divergence(self):
        """All-one curves should give 1.0 instability."""
        curves = [[1.0, 1.0, 1.0]]
        assert instability_score(curves) == 1.0

    def test_partial_divergence(self):
        """Partial divergence should give intermediate score."""
        curves = [[0.0, 0.5, 1.0]]
        score = instability_score(curves)
        assert 0.0 < score < 1.0
        assert score == pytest.approx(0.5)

    def test_multiple_curves_averaged(self):
        """Multiple curves should be averaged."""
        curves = [[0.0, 0.0], [1.0, 1.0]]
        assert instability_score(curves) == pytest.approx(0.5)


class TestSensitivityProfile:
    """Test sensitivity profile across perturbation types."""

    def test_multiple_perturbation_types(self):
        """Should compute instability for each perturbation type."""
        clean = [{"Chaotic": "YES"}, {"Chaotic": "YES"}]
        perturbed = {
            "paraphrase": [{"Chaotic": "YES"}, {"Chaotic": "YES"}],
            "reorder": [{"Chaotic": "YES"}, {"Chaotic": "NO"}],
        }
        profile = sensitivity_profile(clean, perturbed)
        assert "paraphrase" in profile
        assert "reorder" in profile
        assert profile["paraphrase"] == 0.0
        assert profile["reorder"] > 0.0

    def test_empty_perturbed_runs(self):
        """Empty perturbed runs should produce empty profile."""
        clean = [{"Chaotic": "YES"}]
        profile = sensitivity_profile(clean, {})
        assert profile == {}


class TestBeliefFlipRate:
    """Test belief flip rate computation."""

    def test_no_flips(self):
        """Identical beliefs should have 0 flip rate."""
        b = {"Chaotic": "YES", "Random": "NO"}
        assert belief_flip_rate(b, b) == 0.0

    def test_all_flips(self):
        """All different should have 1.0 flip rate."""
        b1 = {"Chaotic": "YES", "Random": "NO"}
        b2 = {"Chaotic": "NO", "Random": "YES"}
        assert belief_flip_rate(b1, b2) == 1.0

    def test_partial_flips(self):
        """Some flips should give intermediate rate."""
        b1 = {"Chaotic": "YES", "Random": "NO"}
        b2 = {"Chaotic": "NO", "Random": "NO"}
        assert belief_flip_rate(b1, b2) == 0.5

    def test_empty_beliefs(self):
        """Empty beliefs should return 0.0."""
        assert belief_flip_rate({}, {}) == 0.0


class TestCorrelationInstabilityAccuracy:
    """Test correlation between instability and accuracy failure."""

    def test_insufficient_data(self):
        """Should return None with fewer than 3 data points."""
        assert correlation_instability_accuracy([0.1], [0.9]) is None
        assert correlation_instability_accuracy([0.1, 0.2], [0.9, 0.8]) is None

    def test_perfect_positive_correlation(self):
        """Higher instability with lower accuracy should give positive correlation."""
        instabilities = [0.1, 0.5, 0.9]
        accuracies = [0.9, 0.5, 0.1]
        corr = correlation_instability_accuracy(instabilities, accuracies)
        assert corr is not None
        assert corr > 0.9

    def test_no_variance(self):
        """Constant values should return None."""
        assert correlation_instability_accuracy([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) is None

    def test_mismatched_lengths(self):
        """Mismatched lengths should return None."""
        assert correlation_instability_accuracy([0.1, 0.2, 0.3], [0.9, 0.8]) is None


class TestExtractBeliefVector:
    """Test belief vector extraction from questions and answers."""

    def test_basic_extraction(self):
        """Should extract predicates from questions and normalize answers."""
        questions = ["Is the system chaotic?", "Is it deterministic?"]
        answers = ["FINAL_ANSWER: YES", "FINAL_ANSWER: NO"]
        belief = extract_belief_vector(questions, answers)
        assert belief["Chaotic"] == "YES"
        assert belief["Deterministic"] == "NO"

    def test_unknown_for_unparseable(self):
        """Unparseable answers should produce UNKNOWN."""
        questions = ["Is the system chaotic?"]
        answers = ["I cannot determine this"]
        belief = extract_belief_vector(questions, answers)
        assert belief["Chaotic"] == "UNKNOWN"

    def test_unrecognized_question_skipped(self):
        """Questions that do not map to a predicate should be skipped."""
        questions = ["What is the weather today?"]
        answers = ["FINAL_ANSWER: YES"]
        belief = extract_belief_vector(questions, answers)
        assert belief == {}


class TestExtractBeliefSequence:
    """Test multi-turn belief sequence extraction."""

    def test_multi_turn(self):
        """Should extract one belief vector per turn."""
        questions = [
            ["Is the system chaotic?"],
            ["Is it deterministic?"],
        ]
        answers = [
            ["FINAL_ANSWER: YES"],
            ["FINAL_ANSWER: YES"],
        ]
        seq = extract_belief_sequence(questions, answers)
        assert len(seq) == 2
        assert seq[0]["Chaotic"] == "YES"
        assert seq[1]["Deterministic"] == "YES"
