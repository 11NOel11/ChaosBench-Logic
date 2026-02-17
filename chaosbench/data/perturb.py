"""Perturbation framework for generating test variants with provenance tracking."""

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chaosbench.data.schemas import Question, Dialogue


@dataclass
class PerturbationRecord:
    """Provenance record for a perturbation applied to data.

    Attributes:
        perturbation_type: One of "paraphrase", "reorder", "contradict", "distract".
        target: What was modified (e.g. "question_text", "turns").
        details: Free-form description of the change.
        seed: Random seed used.
    """

    perturbation_type: str
    target: str
    details: str
    seed: int


SYNONYM_MAP = {
    "chaotic": ["exhibits chaos", "displays chaotic behavior", "behaves chaotically"],
    "deterministic": ["governed by deterministic rules", "follows deterministic dynamics"],
    "sensitive": [
        "exhibits sensitive dependence on initial conditions",
        "shows sensitivity to initial conditions",
    ],
    "periodic": ["exhibits periodic behavior", "has periodic orbits"],
    "random": ["exhibits randomness", "is stochastic", "behaves randomly"],
    "quasi-periodic": ["exhibits quasi-periodic behavior", "is quasiperiodic"],
    "positive lyapunov": [
        "has a positive largest Lyapunov exponent",
        "possesses positive Lyapunov exponent",
    ],
    "strange attractor": [
        "possesses a strange attractor",
        "has a fractal attractor",
    ],
    "fixed point": [
        "converges to a fixed point",
        "has a stable fixed point attractor",
    ],
}


def paraphrase(
    question: Question,
    seed: int = 42,
) -> tuple:
    """Paraphrase a question using synonym substitution.

    Args:
        question: Original question.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (modified Question, PerturbationRecord).
    """
    rng = random.Random(seed)
    new_text = question.question_text

    for keyword, synonyms in SYNONYM_MAP.items():
        if keyword in new_text.lower():
            replacement = rng.choice(synonyms)
            import re
            new_text = re.sub(
                re.escape(keyword),
                replacement,
                new_text,
                count=1,
                flags=re.IGNORECASE,
            )
            break

    modified = copy.deepcopy(question)
    modified.question_text = new_text
    modified.metadata["perturbation"] = "paraphrase"

    record = PerturbationRecord(
        perturbation_type="paraphrase",
        target="question_text",
        details=f"original: {question.question_text!r}, modified: {new_text!r}",
        seed=seed,
    )
    return modified, record


def reorder_premises(
    dialogue: Dialogue,
    seed: int = 42,
) -> tuple:
    """Shuffle the order of turns in a dialogue.

    Args:
        dialogue: Original dialogue.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (modified Dialogue, PerturbationRecord).
    """
    rng = random.Random(seed)
    modified = copy.deepcopy(dialogue)
    original_order = [t.item_id for t in modified.turns]
    rng.shuffle(modified.turns)
    new_order = [t.item_id for t in modified.turns]
    modified.perturbation_type = "reorder"

    record = PerturbationRecord(
        perturbation_type="reorder",
        target="turns",
        details=f"original order: {original_order}, new order: {new_order}",
        seed=seed,
    )
    return modified, record


def inject_contradiction(
    dialogue: Dialogue,
    strength: str = "low",
    seed: int = 42,
) -> tuple:
    """Flip predicate claims in a dialogue to inject contradiction.

    Args:
        dialogue: Original dialogue.
        strength: "low" (flip 1 turn), "med" (flip ~half), "high" (flip all).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (modified Dialogue, PerturbationRecord).
    """
    rng = random.Random(seed)
    modified = copy.deepcopy(dialogue)

    n_turns = len(modified.turns)
    if n_turns == 0:
        record = PerturbationRecord(
            perturbation_type="contradict",
            target="turns",
            details="no turns to flip",
            seed=seed,
        )
        return modified, record

    if strength == "low":
        n_flip = 1
    elif strength == "med":
        n_flip = max(1, n_turns // 2)
    else:
        n_flip = n_turns

    indices = rng.sample(range(n_turns), min(n_flip, n_turns))
    flipped = []

    for idx in indices:
        turn = modified.turns[idx]
        if turn.ground_truth == "YES":
            turn.ground_truth = "NO"
        else:
            turn.ground_truth = "YES"
        turn.metadata["flipped"] = True
        flipped.append(idx)

    modified.perturbation_type = "contradict"

    record = PerturbationRecord(
        perturbation_type="contradict",
        target="turns",
        details=f"strength={strength}, flipped indices: {flipped}",
        seed=seed,
    )
    return modified, record


DISTRACTOR_TEMPLATES = [
    "The system has {n} degrees of freedom.",
    "The initial conditions are drawn from a uniform distribution.",
    "The simulation uses a timestep of {dt}.",
    "The system was first studied in {year}.",
    "The phase space is {dim}-dimensional.",
]


def add_distractors(
    dialogue: Dialogue,
    k: int = 1,
    seed: int = 42,
) -> tuple:
    """Insert plausible but irrelevant distractor turns into a dialogue.

    Args:
        dialogue: Original dialogue.
        k: Number of distractor turns to add.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (modified Dialogue, PerturbationRecord).
    """
    rng = random.Random(seed)
    modified = copy.deepcopy(dialogue)

    added_texts = []
    for i in range(k):
        template = rng.choice(DISTRACTOR_TEMPLATES)
        text = template.format(
            n=rng.randint(2, 10),
            dt=rng.choice(["0.01", "0.001", "0.1"]),
            year=rng.randint(1960, 2020),
            dim=rng.randint(2, 6),
        )

        distractor = Question(
            item_id=f"{dialogue.dialogue_id}_distractor_{i}",
            question_text=text,
            system_id=dialogue.system_id,
            task_family="distractor",
            ground_truth="YES",
            metadata={"is_distractor": True},
        )

        insert_pos = rng.randint(0, len(modified.turns))
        modified.turns.insert(insert_pos, distractor)
        added_texts.append(text)

    modified.perturbation_type = "distract"

    record = PerturbationRecord(
        perturbation_type="distract",
        target="turns",
        details=f"added {k} distractors: {added_texts}",
        seed=seed,
    )
    return modified, record
