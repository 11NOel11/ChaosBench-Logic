"""Data schemas for ChaosBench-Logic v2."""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class SystemInstance:
    """A dynamical system definition.

    Attributes:
        system_id: Unique identifier (e.g. "lorenz63").
        name: Human-readable name.
        category: System category (e.g. "ode", "map", "pde").
        equations: LaTeX or string representation of governing equations.
        parameters: Dict of parameter names to values.
        truth_assignment: Ground truth predicate values {predicate: bool}.
        indicator_labels: Optional dict of chaos indicator results.
    """

    system_id: str
    name: str
    category: str
    equations: str
    parameters: Dict[str, Any]
    truth_assignment: Dict[str, bool]
    indicator_labels: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Check required fields and predicate completeness.

        Returns:
            List of validation error strings (empty if valid).
        """
        errors: List[str] = []
        if not self.system_id:
            errors.append("system_id is required")
        if not self.name:
            errors.append("name is required")
        if not self.truth_assignment:
            errors.append("truth_assignment is required")

        expected_predicates = {
            "Chaotic", "Deterministic", "PosLyap", "Sensitive",
            "StrangeAttr", "PointUnpredictable", "StatPredictable",
            "QuasiPeriodic", "Random", "FixedPointAttr", "Periodic",
        }
        missing = expected_predicates - set(self.truth_assignment.keys())
        if missing:
            errors.append(f"missing predicates: {sorted(missing)}")
        return errors


@dataclass
class Question:
    """A single benchmark question.

    Attributes:
        item_id: Unique item identifier.
        question_text: The question string.
        system_id: Which system this question is about.
        task_family: Task category (e.g. "atomic", "multi_hop", "bias").
        ground_truth: Expected answer ("YES" or "NO").
        predicates: List of predicates this question tests.
        metadata: Additional metadata dict.
    """

    item_id: str
    question_text: str
    system_id: str
    task_family: str
    ground_truth: str
    predicates: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Check required fields.

        Returns:
            List of validation error strings.
        """
        errors: List[str] = []
        if not self.item_id:
            errors.append("item_id is required")
        if not self.question_text:
            errors.append("question_text is required")
        if self.ground_truth not in ("YES", "NO"):
            errors.append(f"ground_truth must be YES or NO, got {self.ground_truth}")
        return errors


@dataclass
class Dialogue:
    """A multi-turn dialogue about a dynamical system.

    Attributes:
        dialogue_id: Unique dialogue identifier.
        system_id: Which system this dialogue is about.
        turns: Ordered list of Questions comprising the dialogue.
        perturbation_type: Optional perturbation applied to this dialogue.
    """

    dialogue_id: str
    system_id: str
    turns: List[Question] = field(default_factory=list)
    perturbation_type: Optional[str] = None

    def validate(self) -> List[str]:
        """Check dialogue validity.

        Returns:
            List of validation error strings.
        """
        errors: List[str] = []
        if not self.dialogue_id:
            errors.append("dialogue_id is required")
        if not self.turns:
            errors.append("dialogue must have at least one turn")
        for i, turn in enumerate(self.turns):
            turn_errors = turn.validate()
            for e in turn_errors:
                errors.append(f"turn {i}: {e}")
        return errors


@dataclass
class AnswerKey:
    """Ground truth answer with explanation.

    Attributes:
        item_id: Matches Question.item_id.
        ground_truth: Expected answer ("YES" or "NO").
        predicate: Primary predicate being tested.
        explanation: Reasoning for the ground truth.
    """

    item_id: str
    ground_truth: str
    predicate: str
    explanation: str

    def validate(self) -> List[str]:
        """Check answer key validity.

        Returns:
            List of validation error strings.
        """
        errors: List[str] = []
        if not self.item_id:
            errors.append("item_id is required")
        if self.ground_truth not in ("YES", "NO"):
            errors.append(f"ground_truth must be YES or NO, got {self.ground_truth}")
        return errors


@dataclass
class DatasetConfig:
    """Dataset version and split configuration.

    Attributes:
        version: Dataset version string.
        splits: Dict mapping split names to file paths.
        content_hash: SHA-256 hash of dataset content for integrity checking.
    """

    version: str
    splits: Dict[str, str] = field(default_factory=dict)
    content_hash: str = ""

    def compute_hash(self, data: str) -> str:
        """Compute SHA-256 hash of data string.

        Args:
            data: String to hash.

        Returns:
            Hex digest string.
        """
        self.content_hash = hashlib.sha256(data.encode("utf-8")).hexdigest()
        return self.content_hash

    def validate(self) -> List[str]:
        """Check config validity.

        Returns:
            List of validation error strings.
        """
        errors: List[str] = []
        if not self.version:
            errors.append("version is required")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict.

        Returns:
            Dict representation.
        """
        return asdict(self)
