"""Quality gate checks for dataset validation.

Implements four quality gates:
1. Near-Duplicate Detection - hash-based exact + Jaccard fuzzy matching
2. Label Leakage Scan - forbidden tokens in question text
3. Class Balance - per-family and per-split YES/NO balance
4. Difficulty Distribution - heuristic scoring (report only)
"""

import hashlib
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class GateResult:
    """Result of a quality gate check.

    Attributes:
        gate_name: Name of the quality gate.
        passed: Whether the gate passed.
        details: Human-readable description of results.
        violations: List of specific violations found.
        stats: Additional statistics dict.
    """
    gate_name: str
    passed: bool
    details: str
    violations: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def _text_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized text."""
    norm = _normalize_text(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two normalized texts (word-level)."""
    words_a = set(_normalize_text(text_a).split())
    words_b = set(_normalize_text(text_b).split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def check_near_duplicates(
    items: List[Dict[str, Any]],
    jaccard_threshold: float = 0.85,
    max_allowed: int = 50,
    exclude_families: Optional[Set[str]] = None,
) -> GateResult:
    """Gate 1: Near-duplicate detection.

    Phase 1: Hash-based exact duplicate check (O(n)).
    Phase 2: Jaccard fuzzy matching on non-duplicates (sampled for performance).

    Args:
        items: List of question dicts with 'question' and 'type' fields.
        jaccard_threshold: Similarity threshold for near-duplicates.
        max_allowed: Maximum allowed near-duplicates before failing.
        exclude_families: Families to exclude from duplicate checking
            (e.g. consistency_paraphrase which intentionally duplicates).

    Returns:
        GateResult with pass/fail and duplicate details.
    """
    if exclude_families is None:
        exclude_families = {"consistency_paraphrase"}

    # Filter to checkable items
    checkable = [
        item for item in items
        if item.get("type", "") not in exclude_families
    ]

    # Phase 1: Exact duplicates via hash
    seen_hashes: Dict[str, List[str]] = defaultdict(list)
    for item in checkable:
        text = item.get("question", "")
        h = _text_hash(text)
        seen_hashes[h].append(item.get("id", "unknown"))

    exact_duplicates = []
    for h, ids in seen_hashes.items():
        if len(ids) > 1:
            exact_duplicates.append(ids)

    n_exact = sum(len(ids) - 1 for ids in exact_duplicates)

    # Phase 2: Jaccard fuzzy matching (sample-based for performance)
    # Only check if we have a manageable number of items
    near_duplicates: List[Tuple[str, str, float]] = []
    if len(checkable) <= 5000:
        texts = [(item.get("id", ""), item.get("question", "")) for item in checkable]
        for i in range(len(texts)):
            for j in range(i + 1, min(i + 100, len(texts))):  # Window-based check
                sim = _jaccard_similarity(texts[i][1], texts[j][1])
                if sim >= jaccard_threshold:
                    # Check it's not an exact duplicate (already counted)
                    h_i = _text_hash(texts[i][1])
                    h_j = _text_hash(texts[j][1])
                    if h_i != h_j:
                        near_duplicates.append((texts[i][0], texts[j][0], sim))

    total_dups = n_exact + len(near_duplicates)
    passed = total_dups <= max_allowed

    violations = []
    for ids in exact_duplicates[:5]:
        violations.append(f"exact duplicate: {ids}")
    for id_a, id_b, sim in near_duplicates[:5]:
        violations.append(f"near-duplicate (Jaccard={sim:.3f}): {id_a} ~ {id_b}")

    return GateResult(
        gate_name="near_duplicate_detection",
        passed=passed,
        details=f"{n_exact} exact duplicates, {len(near_duplicates)} near-duplicates "
                f"(threshold={jaccard_threshold}, max_allowed={max_allowed})",
        violations=violations,
        stats={
            "exact_duplicates": n_exact,
            "near_duplicates": len(near_duplicates),
            "total_duplicates": total_dups,
            "items_checked": len(checkable),
        },
    )


# Forbidden tokens that indicate label leakage
FORBIDDEN_TOKENS = [
    "ground_truth",
    "answer_is",
    "correct answer",
    "the answer is true",
    "the answer is false",
    "the answer is yes",
    "the answer is no",
]


def check_label_leakage(
    items: List[Dict[str, Any]],
    max_allowed: int = 0,
) -> GateResult:
    """Gate 2: Label leakage scan.

    Checks for forbidden tokens in question text and ground truth
    value appearing verbatim as declarative statement.

    Args:
        items: List of question dicts.
        max_allowed: Maximum allowed leaks (default: 0).

    Returns:
        GateResult with pass/fail and leak details.
    """
    violations = []

    for item in items:
        question = item.get("question", "").lower()
        item_id = item.get("id", "unknown")
        gt = item.get("ground_truth", "")

        # Check forbidden tokens
        for token in FORBIDDEN_TOKENS:
            if token in question:
                violations.append(
                    f"{item_id}: forbidden token '{token}' in question text"
                )

        # Check if ground truth appears as declarative statement
        if gt == "TRUE" and "this is true" in question:
            violations.append(f"{item_id}: ground truth 'TRUE' echoed in question")
        if gt == "FALSE" and "this is false" in question:
            violations.append(f"{item_id}: ground truth 'FALSE' echoed in question")

    passed = len(violations) <= max_allowed

    return GateResult(
        gate_name="label_leakage_scan",
        passed=passed,
        details=f"{len(violations)} leaks found (max_allowed={max_allowed})",
        violations=violations[:10],
        stats={"total_leaks": len(violations), "items_checked": len(items)},
    )


def check_class_balance(
    items: List[Dict[str, Any]],
    balance_range: Tuple[float, float] = (0.35, 0.65),
    min_items: int = 20,
) -> GateResult:
    """Gate 3: Class balance check.

    Checks YES/NO (TRUE/FALSE) balance per family and overall.

    Args:
        items: List of question dicts.
        balance_range: Acceptable range for proportion of TRUE/YES labels.
        min_items: Minimum items in a family before balance is checked.

    Returns:
        GateResult with pass/fail and balance details.
    """
    lo, hi = balance_range
    violations = []

    # Group by family
    by_family: Dict[str, List[str]] = defaultdict(list)
    for item in items:
        family = item.get("type", "unknown")
        gt = item.get("ground_truth", "")
        by_family[family].append(gt)

    family_stats: Dict[str, Dict[str, Any]] = {}
    for family, labels in sorted(by_family.items()):
        n = len(labels)
        n_true = sum(1 for l in labels if l in ("TRUE", "YES"))
        ratio = n_true / n if n > 0 else 0.5
        family_stats[family] = {
            "total": n,
            "true_count": n_true,
            "true_ratio": ratio,
        }

        if n >= min_items and (ratio < lo or ratio > hi):
            violations.append(
                f"family '{family}': {ratio:.1%} TRUE ({n_true}/{n}), "
                f"outside [{lo:.0%}, {hi:.0%}]"
            )

    # Overall balance
    all_labels = [item.get("ground_truth", "") for item in items]
    n_total = len(all_labels)
    n_true = sum(1 for l in all_labels if l in ("TRUE", "YES"))
    overall_ratio = n_true / n_total if n_total > 0 else 0.5

    passed = len(violations) == 0

    return GateResult(
        gate_name="class_balance",
        passed=passed,
        details=f"overall balance: {overall_ratio:.1%} TRUE, "
                f"{len(violations)} families out of range",
        violations=violations,
        stats={
            "overall_true_ratio": overall_ratio,
            "per_family": family_stats,
            "items_checked": n_total,
        },
    )


def _compute_difficulty_score(item: Dict[str, Any]) -> float:
    """Compute heuristic difficulty score for a question.

    Factors:
    - Chain/hop length (from metadata)
    - Number of predicates
    - Presence of negation in question text
    - Question type complexity

    Returns:
        Float score from 0.0 (easy) to 1.0 (hard).
    """
    score = 0.0
    question = item.get("question", "")

    # Chain length / hop count
    # (metadata may not be present in JSONL, so check both)
    hop_count = 0
    if "3-hop" in question or "three" in question.lower():
        hop_count = 3
    elif "2-hop" in question:
        hop_count = 2

    # Predicate count (approximate from question text)
    predicate_keywords = [
        "chaotic", "deterministic", "periodic", "sensitive",
        "lyapunov", "attractor", "predictable", "random",
        "quasi-periodic", "fixed point",
    ]
    pred_count = sum(1 for kw in predicate_keywords if kw in question.lower())

    # Negation presence
    negation_patterns = ["not ", "n't ", "false that", "incorrect that", "cannot", "lacks"]
    has_negation = any(p in question.lower() for p in negation_patterns)

    # Type complexity
    item_type = item.get("type", "")
    type_scores = {
        "atomic": 0.1,
        "extended_systems": 0.1,
        "indicator_diagnostic": 0.3,
        "cross_indicator": 0.5,
        "consistency_paraphrase": 0.2,
        "perturbation": 0.3,
        "fol_inference": 0.5,
        "multi_hop": 0.6,
        "adversarial_misleading": 0.7,
        "adversarial_nearmiss": 0.8,
        "adversarial_confusion": 0.7,
        "regime_transition": 0.5,
    }

    score = type_scores.get(item_type, 0.3)
    score += min(0.2, hop_count * 0.1)
    score += min(0.15, pred_count * 0.05)
    if has_negation:
        score += 0.1

    return min(1.0, score)


def check_difficulty_distribution(
    items: List[Dict[str, Any]],
    split_field: str = "_split",
) -> GateResult:
    """Gate 4: Difficulty distribution (report only, never fails).

    Computes heuristic difficulty scores and checks that hard split
    items score higher than core items on average.

    Args:
        items: List of question dicts.
        split_field: Field name for split assignment.

    Returns:
        GateResult (always passes, report only).
    """
    by_split: Dict[str, List[float]] = defaultdict(list)
    all_scores = []

    for item in items:
        score = _compute_difficulty_score(item)
        all_scores.append(score)
        split = item.get(split_field, "unknown")
        by_split[split].append(score)

    split_avgs = {}
    for split, scores in sorted(by_split.items()):
        if scores:
            split_avgs[split] = sum(scores) / len(scores)

    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0

    # Check if hard > core (advisory only)
    hard_avg = split_avgs.get("hard", 0)
    core_avg = split_avgs.get("core", 0)
    ordering_ok = hard_avg >= core_avg or "hard" not in split_avgs

    details = f"overall avg difficulty: {overall_avg:.3f}"
    if split_avgs:
        details += ", per-split: " + ", ".join(
            f"{k}={v:.3f}" for k, v in sorted(split_avgs.items())
        )

    return GateResult(
        gate_name="difficulty_distribution",
        passed=True,  # Report only, never fails
        details=details,
        violations=[f"hard ({hard_avg:.3f}) < core ({core_avg:.3f})"] if not ordering_ok else [],
        stats={
            "overall_avg": overall_avg,
            "per_split_avg": split_avgs,
            "hard_gt_core": ordering_ok,
            "items_scored": len(all_scores),
        },
    )


def run_all_gates(
    items: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> List[GateResult]:
    """Run all quality gates on a dataset.

    Args:
        items: List of question dicts.
        config: Optional quality gate configuration dict with keys:
            - max_near_duplicates: int
            - max_label_leaks: int
            - class_balance_range: [float, float]
            - min_items_for_balance_check: int
            - near_duplicate_jaccard_threshold: float

    Returns:
        List of GateResult objects.
    """
    if config is None:
        config = {}

    results = []

    # Gate 1: Near-duplicate detection
    results.append(check_near_duplicates(
        items,
        jaccard_threshold=config.get("near_duplicate_jaccard_threshold", 0.85),
        max_allowed=config.get("max_near_duplicates", 50),
    ))

    # Gate 2: Label leakage scan
    results.append(check_label_leakage(
        items,
        max_allowed=config.get("max_label_leaks", 0),
    ))

    # Gate 3: Class balance
    balance_range = config.get("class_balance_range", [0.35, 0.65])
    results.append(check_class_balance(
        items,
        balance_range=tuple(balance_range),
        min_items=config.get("min_items_for_balance_check", 20),
    ))

    # Gate 4: Difficulty distribution (report only)
    results.append(check_difficulty_distribution(items))

    return results
