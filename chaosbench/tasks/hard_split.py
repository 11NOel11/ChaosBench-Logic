"""Hard split identification and analysis for ChaosBench-Logic v2."""

import json
import os
from collections import Counter
from typing import Any, Dict, List

from chaosbench.data.schemas import Question


def identify_hard_items(
    results_dir: str = "published_results",
    threshold: float = 0.6,
) -> List[str]:
    """Identify hard task families where accuracy < threshold across models.

    Reads summary.json from each model subdirectory in results_dir and
    averages task_accuracy across all models. Returns task families where
    the average accuracy falls below the threshold.

    Args:
        results_dir: Path to directory containing model result subdirectories.
        threshold: Accuracy threshold below which a task family is considered hard.

    Returns:
        Sorted list of task family names that are hard across models.
    """
    if not os.path.isdir(results_dir):
        return []

    family_scores: Dict[str, List[float]] = {}

    for entry in sorted(os.listdir(results_dir)):
        subdir = os.path.join(results_dir, entry)
        if not os.path.isdir(subdir):
            continue

        summary_path = os.path.join(subdir, "summary.json")
        if not os.path.isfile(summary_path):
            continue

        with open(summary_path, "r") as f:
            summary = json.load(f)

        task_accuracy = summary.get("task_accuracy", {})
        for family, acc in task_accuracy.items():
            if family not in family_scores:
                family_scores[family] = []
            family_scores[family].append(acc)

    hard_families = []
    for family, scores in family_scores.items():
        avg = sum(scores) / len(scores)
        if avg < threshold:
            hard_families.append(family)

    return sorted(hard_families)


def create_hard_split(
    all_items: List[Question],
    hard_families: List[str],
) -> List[Question]:
    """Filter items belonging to hard task families.

    Args:
        all_items: Complete list of benchmark questions.
        hard_families: List of task family names considered hard.

    Returns:
        List of questions whose task_family is in hard_families.
    """
    hard_set = set(hard_families)
    return [q for q in all_items if q.task_family in hard_set]


def analyze_hard_characteristics(
    hard: List[Question],
    all_items: List[Question],
) -> Dict[str, Any]:
    """Analyze characteristics of hard vs all items.

    Computes task family distribution, predicate distribution, and average
    question length for both the hard subset and the full item set.

    Args:
        hard: List of hard questions.
        all_items: Complete list of benchmark questions.

    Returns:
        Dict with keys "hard" and "all", each containing task_family_dist,
        predicate_dist, and avg_question_length.
    """
    def _compute_stats(items: List[Question]) -> Dict[str, Any]:
        """Compute distribution stats for a list of questions.

        Args:
            items: List of Question objects.

        Returns:
            Dict with task_family_dist, predicate_dist, avg_question_length.
        """
        if not items:
            return {
                "task_family_dist": {},
                "predicate_dist": {},
                "avg_question_length": 0.0,
                "count": 0,
            }

        family_counts = Counter(q.task_family for q in items)
        n = len(items)
        family_dist = {k: round(v / n, 4) for k, v in sorted(family_counts.items())}

        pred_counts: Counter = Counter()
        for q in items:
            for p in q.predicates:
                pred_counts[p] += 1
        total_preds = sum(pred_counts.values()) or 1
        pred_dist = {
            k: round(v / total_preds, 4) for k, v in sorted(pred_counts.items())
        }

        avg_len = sum(len(q.question_text) for q in items) / n

        return {
            "task_family_dist": family_dist,
            "predicate_dist": pred_dist,
            "avg_question_length": round(avg_len, 2),
            "count": n,
        }

    return {
        "hard": _compute_stats(hard),
        "all": _compute_stats(all_items),
    }
