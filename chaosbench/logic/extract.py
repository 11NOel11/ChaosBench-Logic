"""Extract predicate vectors from model outputs for belief-state tracking."""

from typing import Dict, List, Optional

from chaosbench.logic.ontology import KEYWORD_MAP
from chaosbench.eval.metrics import normalize_label


def extract_predicate_from_question(question: str) -> Optional[str]:
    """Extract the logical predicate being queried from a question.

    Args:
        question: The question text.

    Returns:
        Predicate name (e.g. "Chaotic", "Deterministic") or None if unknown.
    """
    if not question:
        return None

    q_lower = question.lower()

    for keywords, predicate in KEYWORD_MAP:
        if any(kw in q_lower for kw in keywords):
            return predicate

    return None


def extract_belief_vector(
    questions: List[str],
    answers: List[str],
) -> Dict[str, str]:
    """Parse model outputs into a predicate belief vector for one turn.

    Args:
        questions: List of question texts for this turn.
        answers: List of corresponding model answers.

    Returns:
        Dict mapping predicate names to "YES"/"NO"/"UNKNOWN" for each
        predicate that could be extracted from the questions.
    """
    belief: Dict[str, str] = {}

    for question, answer in zip(questions, answers):
        predicate = extract_predicate_from_question(question)
        if predicate is None:
            continue

        normalized = normalize_label(answer)
        if normalized is not None:
            belief[predicate] = normalized
        else:
            belief[predicate] = "UNKNOWN"

    return belief


def extract_belief_sequence(
    dialogue_questions: List[List[str]],
    dialogue_answers: List[List[str]],
) -> List[Dict[str, str]]:
    """Extract belief vectors for each turn in a multi-turn dialogue.

    Args:
        dialogue_questions: List of question lists, one per turn.
        dialogue_answers: List of answer lists, one per turn.

    Returns:
        List of belief vectors, one per turn.
    """
    sequence: List[Dict[str, str]] = []
    for questions, answers in zip(dialogue_questions, dialogue_answers):
        belief = extract_belief_vector(questions, answers)
        sequence.append(belief)
    return sequence
