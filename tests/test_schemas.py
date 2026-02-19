"""Tests for chaosbench.data.schemas -- data schema validation and construction."""

import pytest

from chaosbench.data.schemas import (
    SystemInstance,
    Question,
    Dialogue,
    AnswerKey,
    DatasetConfig,
)


class TestSystemInstance:
    """Test SystemInstance dataclass."""

    def test_valid_system(self):
        """Valid system should pass validation."""
        s = SystemInstance(
            system_id="lorenz63",
            name="Lorenz-63",
            category="ode",
            equations="dx/dt = sigma*(y - x)",
            parameters={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
            truth_assignment={
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
        )
        assert s.validate() == []

    def test_missing_system_id(self):
        """Missing system_id should be flagged."""
        s = SystemInstance(
            system_id="",
            name="Test",
            category="ode",
            equations="",
            parameters={},
            truth_assignment={"Chaotic": True},
        )
        errors = s.validate()
        assert any("system_id" in e for e in errors)

    def test_missing_predicates(self):
        """Incomplete truth_assignment should list missing predicates."""
        s = SystemInstance(
            system_id="test",
            name="Test",
            category="ode",
            equations="",
            parameters={},
            truth_assignment={"Chaotic": True, "Deterministic": True},
        )
        errors = s.validate()
        assert any("missing predicates" in e for e in errors)

    def test_indicator_labels_default_empty(self):
        """indicator_labels should default to empty dict."""
        s = SystemInstance(
            system_id="test",
            name="Test",
            category="ode",
            equations="",
            parameters={},
            truth_assignment={},
        )
        assert s.indicator_labels == {}


class TestQuestion:
    """Test Question dataclass."""

    def test_valid_question(self):
        """Valid question should pass validation."""
        q = Question(
            item_id="q1",
            question_text="Is the system chaotic?",
            system_id="lorenz63",
            task_family="atomic",
            ground_truth="YES",
            predicates=["Chaotic"],
        )
        assert q.validate() == []

    def test_invalid_ground_truth(self):
        """ground_truth must be YES or NO."""
        q = Question(
            item_id="q1",
            question_text="test",
            system_id="lorenz63",
            task_family="atomic",
            ground_truth="MAYBE",
        )
        errors = q.validate()
        assert any("ground_truth" in e for e in errors)

    def test_missing_item_id(self):
        """Empty item_id should be flagged."""
        q = Question(
            item_id="",
            question_text="test",
            system_id="lorenz63",
            task_family="atomic",
            ground_truth="YES",
        )
        errors = q.validate()
        assert any("item_id" in e for e in errors)


class TestDialogue:
    """Test Dialogue dataclass."""

    def test_valid_dialogue(self):
        """Valid dialogue should pass validation."""
        d = Dialogue(
            dialogue_id="d1",
            system_id="lorenz63",
            turns=[
                Question(
                    item_id="q1",
                    question_text="Is it chaotic?",
                    system_id="lorenz63",
                    task_family="atomic",
                    ground_truth="YES",
                ),
            ],
        )
        assert d.validate() == []

    def test_empty_turns(self):
        """Dialogue with no turns should be flagged."""
        d = Dialogue(dialogue_id="d1", system_id="lorenz63", turns=[])
        errors = d.validate()
        assert any("at least one turn" in e for e in errors)

    def test_invalid_turn_propagates(self):
        """Invalid turn should propagate errors."""
        d = Dialogue(
            dialogue_id="d1",
            system_id="lorenz63",
            turns=[
                Question(
                    item_id="",
                    question_text="test",
                    system_id="lorenz63",
                    task_family="atomic",
                    ground_truth="YES",
                ),
            ],
        )
        errors = d.validate()
        assert any("turn 0" in e for e in errors)


class TestAnswerKey:
    """Test AnswerKey dataclass."""

    def test_valid_answer_key(self):
        """Valid answer key should pass validation."""
        ak = AnswerKey(
            item_id="q1",
            ground_truth="YES",
            predicate="Chaotic",
            explanation="Lorenz-63 is a well-known chaotic system.",
        )
        assert ak.validate() == []

    def test_invalid_ground_truth(self):
        """Invalid ground_truth value should be flagged."""
        ak = AnswerKey(
            item_id="q1",
            ground_truth="UNKNOWN",
            predicate="Chaotic",
            explanation="test",
        )
        errors = ak.validate()
        assert any("ground_truth" in e for e in errors)


class TestDatasetConfig:
    """Test DatasetConfig dataclass."""

    def test_compute_hash(self):
        """Hash should be deterministic."""
        cfg = DatasetConfig(version="2.0.0")
        h1 = cfg.compute_hash("test data")
        h2 = cfg.compute_hash("test data")
        assert h1 == h2
        assert len(h1) == 64

    def test_different_data_different_hash(self):
        """Different data should produce different hash."""
        cfg = DatasetConfig(version="2.0.0")
        h1 = cfg.compute_hash("data1")
        h2 = cfg.compute_hash("data2")
        assert h1 != h2

    def test_validate_requires_version(self):
        """Missing version should be flagged."""
        cfg = DatasetConfig(version="")
        errors = cfg.validate()
        assert any("version" in e for e in errors)

    def test_to_dict(self):
        """to_dict should return serializable dict."""
        cfg = DatasetConfig(version="2.0.0", splits={"train": "train.jsonl"})
        d = cfg.to_dict()
        assert d["version"] == "2.0.0"
        assert "train" in d["splits"]
