"""Grouping logic for perturbation clusters and duplicate detection.

Implements stable group_id computation for questions that should be evaluated
as a cluster (e.g., paraphrases, perturbations of the same base question).
"""

import hashlib
import re
from typing import Any, Dict, Optional


def compute_group_id(item: Dict[str, Any]) -> Optional[str]:
    """Compute stable group_id for perturbation clusters.

    For perturbation families (paraphrase, distractor, negation, entity_swap),
    derives group_id from base question signature. Items with the same group_id
    should be evaluated together to measure flip rates.

    Args:
        item: Question dict with fields: id, question, type, system_id, ground_truth.

    Returns:
        group_id string, or None if not part of a perturbation group.
    """
    item_type = item.get("type", "")
    item_id = item.get("id", "")

    # Only apply grouping to perturbation families
    if not item_type.startswith("perturbation"):
        # Atomic and other families are singletons
        return None

    # Extract base question ID from perturbation ID
    # Format: perturb_{variant}_{base_id}
    # E.g.: perturb_paraphrase_0063 → base_id = 0063
    #       perturb_entity_swap_1817 → base_id = 1817
    match = re.match(r"perturb_(\w+)_(\d+)", item_id)
    if not match:
        # Fallback: use hash of canonical question signature
        return _compute_canonical_hash(item)

    variant_type = match.group(1)  # paraphrase, distractor, entity_swap, negation
    base_id = match.group(2)

    # Group by: predicate + system_family + base_id
    # This groups perturbations of the same base question together
    system_id = item.get("system_id", "")
    question = item.get("question", "")

    # Extract predicate from question (lowercase keywords)
    predicate = _extract_predicate(question)

    # System family: core vs dysts
    if system_id.startswith("dysts_"):
        system_family = "dysts"
    else:
        system_family = "core"

    # Group ID format: perturb_{predicate}_{system_family}_{base_id}
    group_id = f"perturb_{predicate}_{system_family}_{base_id}"

    return group_id


def _extract_predicate(question: str) -> str:
    """Extract predicate keyword from question text.

    Args:
        question: Question text.

    Returns:
        Predicate name (lowercase) or "unknown" if not found.
    """
    q_lower = question.lower()

    predicates = [
        "chaotic",
        "deterministic",
        "poslyap",
        "sensitive",
        "strangeattr",
        "pointunpredictable",
        "statpredictable",
        "quasiperiodic",
        "random",
        "fixedpointattr",
        "periodic",
    ]

    for pred in predicates:
        if pred in q_lower:
            return pred

    # Fallback: check for indicator names
    if "lyapunov" in q_lower:
        return "poslyap"
    if "attractor" in q_lower:
        return "strangeattr"

    return "unknown"


def _compute_canonical_hash(item: Dict[str, Any]) -> str:
    """Compute hash-based fallback group_id from question signature.

    Used when perturbation ID doesn't match expected format.

    Args:
        item: Question dict.

    Returns:
        Hash-based group_id.
    """
    system_id = item.get("system_id", "")
    question = item.get("question", "")
    predicate = _extract_predicate(question)

    # Normalize question: remove distractors, lowercase, strip punctuation
    q_norm = re.sub(r"[^\w\s]", "", question.lower())
    q_norm = re.sub(r"\s+", " ", q_norm).strip()

    signature = f"{system_id}:{predicate}:{q_norm[:100]}"
    h = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]

    return f"perturb_hash_{h}"


def is_accidental_duplicate(
    item_a: Dict[str, Any],
    item_b: Dict[str, Any]
) -> bool:
    """Check if two items are accidental duplicates (same question + system + answer).

    Intentional perturbation groups share group_id but may have different IDs.
    Accidental duplicates are identical items with different IDs but no grouping intent.

    Args:
        item_a: First item dict.
        item_b: Second item dict.

    Returns:
        True if accidental duplicate, False if intentional group or distinct.
    """
    # Exact match: question + system_id + ground_truth + type
    return (
        _normalize_text(item_a.get("question", "")) == _normalize_text(item_b.get("question", ""))
        and item_a.get("system_id") == item_b.get("system_id")
        and item_a.get("ground_truth") == item_b.get("ground_truth")
        and item_a.get("type") == item_b.get("type")
        and item_a.get("id") != item_b.get("id")
    )


def _normalize_text(text: str) -> str:
    """Normalize text for duplicate detection: lowercase, strip punctuation, collapse whitespace."""
    import string
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text
