"""Results aggregation utilities."""

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List

from chaosbench.eval.metrics import EvalResult


def aggregate_runs(
    run_dirs: List[str],
) -> Dict[str, Any]:
    """Aggregate summary statistics across multiple evaluation runs.

    Args:
        run_dirs: List of run output directories, each containing summary.json.

    Returns:
        Dict with per-run summaries and aggregate statistics.
    """
    summaries: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        summary_path = os.path.join(run_dir, "summary.json")
        if not os.path.exists(summary_path):
            continue
        with open(summary_path, "r", encoding="utf-8") as f:
            summaries.append(json.load(f))

    if not summaries:
        return {"runs": [], "aggregate": {}}

    accuracies = [
        s["overall_accuracy"]
        for s in summaries
        if s.get("overall_accuracy") is not None
    ]

    aggregate: Dict[str, Any] = {}
    if accuracies:
        aggregate["mean_accuracy"] = sum(accuracies) / len(accuracies)
        aggregate["min_accuracy"] = min(accuracies)
        aggregate["max_accuracy"] = max(accuracies)

    return {"runs": summaries, "aggregate": aggregate}


def results_to_jsonl(results: List[EvalResult]) -> str:
    """Serialize EvalResult list to JSONL string.

    Args:
        results: List of EvalResult objects.

    Returns:
        JSONL-formatted string.
    """
    lines = [json.dumps(asdict(r)) for r in results]
    return "\n".join(lines)
