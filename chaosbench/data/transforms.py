"""Data transforms for converting between formats."""

from typing import Any, Dict, List

from chaosbench.data.schemas import Question, Dialogue


def questions_to_dialogue(
    questions: List[Question],
    dialogue_id: str,
    system_id: str,
) -> Dialogue:
    """Bundle a list of questions into a dialogue.

    Args:
        questions: List of Question objects.
        dialogue_id: Identifier for the new dialogue.
        system_id: System this dialogue is about.

    Returns:
        A Dialogue containing the questions as turns.
    """
    return Dialogue(
        dialogue_id=dialogue_id,
        system_id=system_id,
        turns=list(questions),
    )


def jsonl_item_to_question(item: Dict[str, Any]) -> Question:
    """Convert a raw JSONL item dict to a Question schema.

    Args:
        item: Raw dict from JSONL file.

    Returns:
        A Question dataclass instance.
    """
    ground_truth_raw = (
        item.get("ground_truth")
        or item.get("gold_label")
        or item.get("gold")
        or item.get("answer")
        or item.get("label")
        or "YES"
    )
    gt = ground_truth_raw.upper() if isinstance(ground_truth_raw, str) else "YES"
    if gt not in ("YES", "NO"):
        gt = "YES"

    return Question(
        item_id=item.get("id", ""),
        question_text=item.get("question", ""),
        system_id=item.get("system_id", ""),
        task_family=item.get("task_family") or item.get("type") or "unknown",
        ground_truth=gt,
        predicates=item.get("predicates", []),
        metadata={
            k: v
            for k, v in item.items()
            if k not in ("id", "question", "system_id", "task_family", "type",
                         "ground_truth", "gold_label", "gold", "answer", "label",
                         "predicates")
        },
    )
