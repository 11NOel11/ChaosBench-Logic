"""Cross-question consistency tasks for testing belief stability."""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

from chaosbench.data.schemas import Question, Dialogue
from chaosbench.data.perturb import paraphrase, reorder_premises
from chaosbench.logic.ontology import PREDICATES


@dataclass
class ConsistencySet:
    """A set of questions expected to produce consistent answers.

    Attributes:
        set_id: Unique identifier for this consistency set.
        system_id: Dynamical system being tested.
        predicate: The predicate all questions target.
        expected_answer: The consistent answer expected.
        questions: List of variant questions.
        set_type: Type of consistency test (e.g. "overlap", "paraphrase", "reorder").
    """

    set_id: str
    system_id: str
    predicate: str
    expected_answer: str
    questions: List[Question] = field(default_factory=list)
    set_type: str = "overlap"


def generate_overlap_set(
    system_id: str,
    predicate: str,
    expected_answer: str,
    base_questions: List[str],
    set_id: str = "",
) -> ConsistencySet:
    """Generate overlapping queries (same predicate, different phrasing).

    Args:
        system_id: Target system.
        predicate: Predicate being tested.
        expected_answer: Expected consistent answer.
        base_questions: List of different phrasings for the same query.
        set_id: Identifier for the set.

    Returns:
        ConsistencySet with all variant questions.
    """
    questions = []
    for i, q_text in enumerate(base_questions):
        q = Question(
            item_id=f"{set_id}_overlap_{i}",
            question_text=q_text,
            system_id=system_id,
            task_family="consistency_overlap",
            ground_truth=expected_answer,
            predicates=[predicate],
        )
        questions.append(q)

    return ConsistencySet(
        set_id=set_id or f"overlap_{system_id}_{predicate}",
        system_id=system_id,
        predicate=predicate,
        expected_answer=expected_answer,
        questions=questions,
        set_type="overlap",
    )


def generate_paraphrase_set(
    base_question: Question,
    n_variants: int = 3,
    seed: int = 42,
) -> ConsistencySet:
    """Generate paraphrased variants of a question.

    Args:
        base_question: Original question to paraphrase.
        n_variants: Number of paraphrased variants to generate.
        seed: Random seed for reproducibility.

    Returns:
        ConsistencySet with original + paraphrased questions.
    """
    questions = [base_question]

    for i in range(n_variants):
        variant, _ = paraphrase(base_question, seed=seed + i)
        variant.item_id = f"{base_question.item_id}_para_{i}"
        variant.task_family = "consistency_paraphrase"
        questions.append(variant)

    predicate = ""
    if base_question.predicates:
        predicate = base_question.predicates[0]

    return ConsistencySet(
        set_id=f"paraphrase_{base_question.item_id}",
        system_id=base_question.system_id,
        predicate=predicate,
        expected_answer=base_question.ground_truth,
        questions=questions,
        set_type="paraphrase",
    )


def generate_reorder_set(
    dialogue: Dialogue,
    n_variants: int = 3,
    seed: int = 42,
) -> List[Dialogue]:
    """Generate premise-reordered variants of a dialogue.

    Args:
        dialogue: Original dialogue.
        n_variants: Number of reordered variants.
        seed: Random seed for reproducibility.

    Returns:
        List of reordered Dialogue variants (including original).
    """
    variants = [dialogue]

    for i in range(n_variants):
        reordered, _ = reorder_premises(dialogue, seed=seed + i)
        reordered.dialogue_id = f"{dialogue.dialogue_id}_reorder_{i}"
        variants.append(reordered)

    return variants


def score_consistency(
    consistency_set: ConsistencySet,
    predictions: Dict[str, str],
) -> Dict[str, Any]:
    """Score model consistency across a set of variant questions.

    Args:
        consistency_set: Set of questions expected to have the same answer.
        predictions: Dict mapping item_id to predicted label ("YES"/"NO").

    Returns:
        Dict with consistency_rate, expected_match_rate, and details.
    """
    answers = []
    for q in consistency_set.questions:
        pred = predictions.get(q.item_id)
        if pred is not None:
            answers.append(pred)

    if not answers:
        return {
            "consistency_rate": None,
            "expected_match_rate": None,
            "n_answered": 0,
        }

    most_common = max(set(answers), key=answers.count)
    consistent = sum(1 for a in answers if a == most_common)
    consistency_rate = consistent / len(answers)

    expected_matches = sum(
        1 for a in answers if a == consistency_set.expected_answer
    )
    expected_match_rate = expected_matches / len(answers)

    return {
        "consistency_rate": consistency_rate,
        "expected_match_rate": expected_match_rate,
        "n_answered": len(answers),
        "most_common_answer": most_common,
    }
