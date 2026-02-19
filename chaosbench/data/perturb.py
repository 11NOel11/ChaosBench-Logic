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


def _apply_structural_transformation(text: str, transform_type: str, rng: random.Random) -> str:
    """Apply structural transformation to question text (v2.2 dedupe fix).

    Transforms sentence structure rather than just substituting synonyms,
    creating variants with lower normalized text similarity.

    Args:
        text: Original question text.
        transform_type: Type of transformation (voice, framing, structure).
        rng: Random number generator.

    Returns:
        Structurally transformed question text.
    """
    import re

    # Extract system name and predicate from common patterns
    # Strategy: match greedily up to the last word (predicate) before "?"
    # "Is the Lorenz system chaotic?" → system="Lorenz system", predicate="chaotic"
    match_is = re.match(r"Is (?:the )?(.+)\s+(\S+)\?$", text, re.IGNORECASE)
    match_does = re.match(r"Does (?:the )?(.+?)\s+(exhibit|display|have|possess)\s+(.+)\?", text, re.IGNORECASE)
    match_would = re.match(r"Would you classify (?:the )?(.+?)\s+as\s+(.+)\?", text, re.IGNORECASE)

    if transform_type == "voice":
        # Active ↔ Passive voice transformation
        if match_is:
            system = match_is.group(1)
            predicate = match_is.group(2)
            # Active: "Is the Lorenz system chaotic?"
            # Passive: "Can the Lorenz system be characterized as chaotic?"
            return f"Can {system} be characterized as {predicate}?"
        elif "characterized" in text.lower():
            # Reverse transformation
            match_char = re.match(r"Can (?:the )?(.+?)\s+be characterized as\s+(.+)\?", text, re.IGNORECASE)
            if match_char:
                system = match_char.group(1)
                predicate = match_char.group(2)
                return f"Is {system} {predicate}?"

    elif transform_type == "framing":
        # Positive ↔ Negative framing
        if match_is:
            system = match_is.group(1)
            predicate = match_is.group(2)
            # Positive: "Is the Lorenz system chaotic?"
            # Negative: "Would it be incorrect to say the Lorenz system is not chaotic?"
            return f"Would it be incorrect to say {system} is not {predicate}?"
        elif "incorrect" in text.lower() and "not" in text.lower():
            # Reverse transformation (double negative → positive)
            match_neg = re.match(r"Would it be incorrect to say (?:the )?(.+?)\s+is not\s+(.+)\?", text, re.IGNORECASE)
            if match_neg:
                system = match_neg.group(1)
                predicate = match_neg.group(2)
                return f"Is {system} {predicate}?"

    elif transform_type == "structure":
        # Question ↔ Statement + verification structure
        if match_is:
            system = match_is.group(1)
            predicate = match_is.group(2)
            # Question: "Is the Lorenz system chaotic?"
            # Statement: "Consider the claim: the Lorenz system is chaotic. Is this claim accurate?"
            return f"Consider the claim: {system} is {predicate}. Is this claim accurate?"
        elif "consider the claim" in text.lower():
            # Reverse transformation
            match_claim = re.match(r"Consider the claim:\s*(?:the )?(.+?)\s+is\s+(.+)\.\s*Is this claim accurate\?", text, re.IGNORECASE)
            if match_claim:
                system = match_claim.group(1)
                predicate = match_claim.group(2)
                return f"Is {system} {predicate}?"

    # Fallback: return original text if no transformation applied
    return text


def paraphrase(
    question: Question,
    seed: int = 42,
) -> tuple:
    """Paraphrase a question using structural transformation (v2.2 improved).

    v2.2 Change: Replaced synonym substitution with structural transformations
    (voice, framing, structure) to reduce normalized text similarity and
    lower dedupe rate from 31.7% to ≤30%.

    Args:
        question: Original question.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (modified Question, PerturbationRecord).
    """
    rng = random.Random(seed)
    original_text = question.question_text

    # Choose transformation type based on seed
    transform_types = ["voice", "framing", "structure"]
    transform_type = transform_types[seed % len(transform_types)]

    # Apply structural transformation
    new_text = _apply_structural_transformation(original_text, transform_type, rng)

    # If no structural transformation applied, fall back to synonym substitution
    if new_text == original_text:
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
    modified.metadata["paraphrase_type"] = transform_type if new_text != original_text else "synonym"

    record = PerturbationRecord(
        perturbation_type="paraphrase",
        target="question_text",
        details=f"type: {transform_type}, original: {original_text!r}, modified: {new_text!r}",
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
