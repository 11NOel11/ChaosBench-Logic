"""Predicate extraction and polarity inference for CARE-v3."""

from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple

from chaosbench.logic.ontology import KEYWORD_MAP

NEGATION_PATTERNS = (
    "is it false that",
    "is it incorrect that",
    "is it untrue that",
    "would it be incorrect",
    "would you say it is incorrect",
    "cannot be characterized",
    "not be characterized",
    " not ",
)


def _iter_keyword_hits(text: str, keyword: str) -> Iterable[int]:
    start = 0
    while True:
        idx = text.find(keyword, start)
        if idx < 0:
            return
        end = idx + len(keyword)
        left_ok = idx == 0 or not text[idx - 1].isalnum()
        right_ok = end >= len(text) or not text[end].isalnum()
        if left_ok and right_ok:
            yield idx
        start = idx + 1


def _tail_clause(question: str) -> str:
    text = question.strip()
    if not text:
        return text

    for sep in ("?", ";", ":"):
        if sep in text:
            text = text.rsplit(sep, 1)[-1].strip() or question

    lower = text.lower()
    for marker in (" is ", " be ", " as "):
        idx = lower.rfind(marker)
        if idx >= 0:
            return text[idx + len(marker) :].strip() or question

    return text


def _first_match(question_lower: str) -> Optional[str]:
    for keywords, predicate in KEYWORD_MAP:
        for keyword in keywords:
            for _ in _iter_keyword_hits(question_lower, keyword.lower()):
                return predicate
    return None


def _last_mention(question_lower: str) -> Optional[str]:
    best: Tuple[int, Optional[str]] = (-1, None)
    for keywords, predicate in KEYWORD_MAP:
        for keyword in keywords:
            for idx in _iter_keyword_hits(question_lower, keyword.lower()):
                if idx > best[0]:
                    best = (idx, predicate)
    return best[1]


def extract_predicate(question: str, strategy: str = "first_match") -> Optional[str]:
    """Extract a predicate name from question text."""
    if not question:
        return None

    q_lower = question.lower()
    if strategy == "first_match":
        return _first_match(q_lower)
    if strategy == "last_mention":
        return _last_mention(q_lower)
    if strategy == "tail_clause":
        tail = _tail_clause(question).lower()
        pred = _last_mention(tail)
        if pred is not None:
            return pred
        return _last_mention(q_lower)

    raise ValueError(f"Unknown extractor strategy: {strategy}")


def infer_polarity(question: str, mode: str = "rule_based") -> int:
    """Infer question polarity: +1 for direct, -1 for negated."""
    if mode == "none":
        return 1
    if mode != "rule_based":
        raise ValueError(f"Unknown polarity mode: {mode}")

    q = f" {question.lower().strip()} "
    if any(pattern in q for pattern in NEGATION_PATTERNS):
        return -1

    if re.search(r"\bis(?:n['’]t)?\s+it\s+false\b", q):
        return -1

    return 1


def label_to_predicate_truth(answer_label: str, polarity: int) -> str:
    """Map answer label (TRUE/FALSE) to predicate truth (YES/NO)."""
    if answer_label not in {"TRUE", "FALSE"}:
        raise ValueError(f"Invalid answer label: {answer_label}")
    if polarity not in {-1, 1}:
        raise ValueError(f"Invalid polarity: {polarity}")

    truth_yes = answer_label == "TRUE"
    if polarity < 0:
        truth_yes = not truth_yes
    return "YES" if truth_yes else "NO"


def predicate_truth_to_label(predicate_truth: str, polarity: int) -> str:
    """Map predicate truth (YES/NO) back to answer label (TRUE/FALSE)."""
    if predicate_truth not in {"YES", "NO"}:
        raise ValueError(f"Invalid predicate truth: {predicate_truth}")
    if polarity not in {-1, 1}:
        raise ValueError(f"Invalid polarity: {polarity}")

    answer_true = predicate_truth == "YES"
    if polarity < 0:
        answer_true = not answer_true
    return "TRUE" if answer_true else "FALSE"
