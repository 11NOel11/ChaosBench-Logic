"""Tests for chaosbench.data.perturb -- perturbation framework."""

import pytest

from chaosbench.data.schemas import Question, Dialogue
from chaosbench.data.perturb import (
    PerturbationRecord,
    paraphrase,
    reorder_premises,
    inject_contradiction,
    add_distractors,
)


def make_question(item_id="q1", text="Is the system chaotic?", gt="YES"):
    """Helper to create a test question."""
    return Question(
        item_id=item_id,
        question_text=text,
        system_id="lorenz63",
        task_family="atomic",
        ground_truth=gt,
        predicates=["Chaotic"],
    )


def make_dialogue(n_turns=3):
    """Helper to create a test dialogue."""
    turns = [
        make_question(f"q{i}", f"Question {i} about the system?", "YES")
        for i in range(n_turns)
    ]
    return Dialogue(
        dialogue_id="d1",
        system_id="lorenz63",
        turns=turns,
    )


class TestParaphrase:
    """Test paraphrase perturbation."""

    def test_returns_modified_question_and_record(self):
        """Should return a (Question, PerturbationRecord) tuple."""
        q = make_question()
        modified, record = paraphrase(q, seed=42)
        assert isinstance(modified, Question)
        assert isinstance(record, PerturbationRecord)
        assert record.perturbation_type == "paraphrase"

    def test_preserves_ground_truth(self):
        """Paraphrase should not change the ground truth."""
        q = make_question(gt="NO")
        modified, _ = paraphrase(q, seed=42)
        assert modified.ground_truth == "NO"

    def test_deterministic_with_seed(self):
        """Same seed should produce same result."""
        q = make_question()
        m1, _ = paraphrase(q, seed=42)
        m2, _ = paraphrase(q, seed=42)
        assert m1.question_text == m2.question_text

    def test_different_seeds_may_differ(self):
        """Different seeds may produce different paraphrases."""
        q = make_question()
        m1, _ = paraphrase(q, seed=42)
        m2, _ = paraphrase(q, seed=99)
        # Not guaranteed to differ but should not crash
        assert isinstance(m1.question_text, str)
        assert isinstance(m2.question_text, str)

    def test_provenance_recorded(self):
        """Perturbation record should contain provenance details."""
        q = make_question()
        _, record = paraphrase(q, seed=42)
        assert record.seed == 42
        assert "original" in record.details
        assert "modified" in record.details


class TestReorderPremises:
    """Test premise reordering perturbation."""

    def test_returns_dialogue_and_record(self):
        """Should return a (Dialogue, PerturbationRecord) tuple."""
        d = make_dialogue(3)
        modified, record = reorder_premises(d, seed=42)
        assert isinstance(modified, Dialogue)
        assert isinstance(record, PerturbationRecord)
        assert record.perturbation_type == "reorder"

    def test_preserves_turn_count(self):
        """Reordering should preserve the number of turns."""
        d = make_dialogue(5)
        modified, _ = reorder_premises(d, seed=42)
        assert len(modified.turns) == 5

    def test_preserves_turn_content(self):
        """All original turns should be present after reorder."""
        d = make_dialogue(4)
        original_ids = {t.item_id for t in d.turns}
        modified, _ = reorder_premises(d, seed=42)
        modified_ids = {t.item_id for t in modified.turns}
        assert original_ids == modified_ids

    def test_deterministic(self):
        """Same seed should produce same reorder."""
        d = make_dialogue(5)
        m1, _ = reorder_premises(d, seed=42)
        m2, _ = reorder_premises(d, seed=42)
        ids1 = [t.item_id for t in m1.turns]
        ids2 = [t.item_id for t in m2.turns]
        assert ids1 == ids2


class TestInjectContradiction:
    """Test contradiction injection perturbation."""

    def test_low_strength_flips_one(self):
        """Low strength should flip exactly 1 turn."""
        d = make_dialogue(5)
        modified, record = inject_contradiction(d, strength="low", seed=42)
        flipped = [t for t in modified.turns if t.metadata.get("flipped")]
        assert len(flipped) == 1
        assert "low" in record.details

    def test_high_strength_flips_all(self):
        """High strength should flip all turns."""
        d = make_dialogue(3)
        modified, _ = inject_contradiction(d, strength="high", seed=42)
        flipped = [t for t in modified.turns if t.metadata.get("flipped")]
        assert len(flipped) == 3

    def test_flipped_turns_change_gt(self):
        """Flipped turns should have inverted ground truth."""
        d = make_dialogue(1)
        d.turns[0].ground_truth = "YES"
        modified, _ = inject_contradiction(d, strength="high", seed=42)
        assert modified.turns[0].ground_truth == "NO"

    def test_empty_dialogue(self):
        """Empty dialogue should not crash."""
        d = Dialogue(dialogue_id="d1", system_id="lorenz63", turns=[])
        modified, record = inject_contradiction(d, strength="low", seed=42)
        assert len(modified.turns) == 0
        assert "no turns" in record.details


class TestAddDistractors:
    """Test distractor injection perturbation."""

    def test_adds_k_distractors(self):
        """Should add exactly k distractor turns."""
        d = make_dialogue(3)
        modified, record = add_distractors(d, k=2, seed=42)
        assert len(modified.turns) == 5
        assert record.perturbation_type == "distract"

    def test_distractors_marked(self):
        """Distractor turns should be marked in metadata."""
        d = make_dialogue(2)
        modified, _ = add_distractors(d, k=1, seed=42)
        distractors = [t for t in modified.turns if t.metadata.get("is_distractor")]
        assert len(distractors) == 1

    def test_deterministic(self):
        """Same seed should produce same distractors."""
        d = make_dialogue(3)
        m1, _ = add_distractors(d, k=2, seed=42)
        m2, _ = add_distractors(d, k=2, seed=42)
        texts1 = [t.question_text for t in m1.turns]
        texts2 = [t.question_text for t in m2.turns]
        assert texts1 == texts2

    def test_original_turns_preserved(self):
        """Original turns should still be present."""
        d = make_dialogue(3)
        original_ids = {t.item_id for t in d.turns}
        modified, _ = add_distractors(d, k=2, seed=42)
        modified_ids = {t.item_id for t in modified.turns}
        assert original_ids.issubset(modified_ids)
