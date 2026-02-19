"""Evaluation runner: data loading, parallel evaluation, checkpointing, exports."""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from chaosbench.models.prompt import ModelConfig, ModelClient
from chaosbench.eval.metrics import EvalResult, normalize_label, compute_summary, Outcome
from chaosbench.data.grouping import compute_group_id

if TYPE_CHECKING:
    from chaosbench.eval.cache import ResponseCache


PROVIDER_POLICIES = {
    "openai": {"max_workers": 5, "delay": 0.1},
    "anthropic": {"max_workers": 4, "delay": 0.15},
    "google": {"max_workers": 8, "delay": 0.05},
    "huggingface": {"max_workers": 3, "delay": 0.15},
    "default": {"max_workers": 2, "delay": 0.2},
}


def get_provider_policy(model_name: str) -> Dict[str, Any]:
    """Infer provider from model name and return rate limiting policy.

    Args:
        model_name: Model identifier string.

    Returns:
        Dict with max_workers and delay keys.
    """
    model_lower = model_name.lower()

    if "gpt" in model_lower or "openai" in model_lower:
        return PROVIDER_POLICIES["openai"]
    elif "claude" in model_lower or "anthropic" in model_lower:
        return PROVIDER_POLICIES["anthropic"]
    elif "gemini" in model_lower or "google" in model_lower:
        return PROVIDER_POLICIES["google"]
    elif any(
        keyword in model_lower
        for keyword in ["llama", "mixtral", "openhermes", "huggingface", "hf"]
    ):
        return PROVIDER_POLICIES["huggingface"]
    else:
        return PROVIDER_POLICIES["default"]


def retry_with_backoff(func, max_retries=4, initial_delay=1.0):
    """Retry a function with exponential backoff.

    Args:
        func: Callable to retry.
        max_retries: Maximum number of retries.
        initial_delay: Initial delay in seconds.

    Returns:
        Tuple of (result, error_type, error_msg).
    """
    last_error = None
    last_error_type = "OtherError"

    for attempt in range(max_retries + 1):
        try:
            result = func()
            return result, None, None
        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            if "429" in error_str or "rate limit" in error_str or "rate_limit" in error_str:
                error_type = "RateLimitError"
                is_retryable = True
            elif (
                "401" in error_str
                or "403" in error_str
                or "authentication" in error_str
                or "unauthorized" in error_str
            ):
                error_type = "AuthError"
                is_retryable = False
            elif (
                "api key" in error_str
                or "api_key" in error_str
                or "invalid x-api-key" in error_str
            ):
                error_type = "InvalidAPIKeyError"
                is_retryable = False
            elif any(code in error_str for code in ["500", "502", "503", "504"]):
                error_type = "ServerError"
                is_retryable = True
            elif "timeout" in error_str:
                error_type = "TimeoutError"
                is_retryable = True
            else:
                error_type = "OtherError"
                is_retryable = False

            last_error_type = error_type

            if not is_retryable:
                return None, error_type, f"{error_type}: {str(e)[:200]}"

            if attempt < max_retries:
                delay = initial_delay * (2 ** attempt)
                print(
                    f"  Retry attempt {attempt + 1}/{max_retries} after {delay}s "
                    f"({error_type}: {str(e)[:100]})"
                )
                time.sleep(delay)
            else:
                return (
                    None,
                    error_type,
                    f"All {max_retries} retries failed - {error_type}: {str(e)[:200]}",
                )

    return None, last_error_type, f"Unexpected retry loop exit: {str(last_error)[:200]}"


def normalize_ground_truth(value: str) -> str:
    """Normalize ground truth to canonical TRUE/FALSE format.

    Accepts legacy v1 format (YES/NO) and v2 format (TRUE/FALSE).
    Passes through non-binary values unchanged for v1 compatibility.

    Args:
        value: Ground truth value.

    Returns:
        Normalized value: "TRUE", "FALSE", or unchanged if non-binary.
    """
    normalized = value.upper().strip()
    if normalized in {"TRUE", "YES", "Y", "T"}:
        return "TRUE"
    elif normalized in {"FALSE", "NO", "N", "F"}:
        return "FALSE"
    else:
        return value


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load records from a JSONL file.

    Normalizes ground_truth field from legacy YES/NO to TRUE/FALSE.

    Args:
        path: Path to JSONL file.

    Returns:
        List of parsed JSON records with normalized ground_truth.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "ground_truth" in record:
                record["ground_truth"] = normalize_ground_truth(record["ground_truth"])
            records.append(record)
    return records


def load_batches(batch_paths: List[str]) -> List[Dict[str, Any]]:
    """Load and merge multiple JSONL batch files.

    Args:
        batch_paths: List of paths to JSONL batch files.

    Returns:
        Merged list of items with _batch_file metadata.
    """
    all_items: List[Dict[str, Any]] = []
    for bp in batch_paths:
        items = load_jsonl(bp)
        base = os.path.basename(bp)
        for item in items:
            item["_batch_file"] = base
        all_items.extend(items)
    return all_items


def evaluate_single_item_robust(
    item: Dict[str, Any],
    client: ModelClient,
    numeric_fact_map: Dict[str, str],
    delay: float = 0.0,
    cache: Optional['ResponseCache'] = None,
    model_name: str = "unknown",
    mode: str = "zeroshot",
) -> EvalResult:
    """Evaluate a single item with retry logic and robust field extraction.

    Args:
        item: Question item from dataset.
        client: Model client.
        numeric_fact_map: System ID to numeric facts mapping.
        delay: Rate limiting delay in seconds.
        cache: Optional ResponseCache for caching responses.
        model_name: Model name for cache key.
        mode: Evaluation mode for cache key.

    Returns:
        EvalResult with correct=None if all retries failed.
    """
    if delay > 0:
        time.sleep(delay)

    q = item.get("question", "")
    item_id = item.get("id", "")
    system_id = item.get("system_id")

    ctx: Dict[str, Any] = {}
    if client.config.mode == "tool" and system_id in numeric_fact_map:
        ctx["numeric_facts"] = numeric_fact_map[system_id]

    gold_raw = (
        item.get("ground_truth")
        or item.get("gold_label")
        or item.get("gold")
        or item.get("answer")
        or item.get("label")
    )
    gold = normalize_label(gold_raw)

    task_family = item.get("task_family") or item.get("type") or "unknown"
    bias_family = item.get("bias_family") or item.get("bias_type") or item.get("bias")
    turn_index = item.get("turn_index") or item.get("turn")

    # Check cache first
    pred_text = None
    error_type = None
    error_msg = None
    if cache is not None:
        pred_text = cache.get(model_name, mode, item_id, q)
        if pred_text is not None:
            # Cache hit - skip model call
            pass

    # Call model if not cached
    if pred_text is None:
        def call_model():
            prompt = client.build_prompt(q, context=ctx)
            return client.call(prompt)

        pred_text, error_type, error_msg = retry_with_backoff(
            call_model, max_retries=4, initial_delay=1.0
        )

        # Store successful responses in cache
        if cache is not None and pred_text is not None:
            cache.put(model_name, mode, item_id, q, pred_text)

    pred_norm = normalize_label(pred_text) if pred_text is not None else None

    if pred_text is not None and pred_norm is None:
        print(
            f"  Warning: normalize_label failed to extract YES/NO from: {pred_text[:100]}..."
        )

    # Compute outcome (3-way)
    outcome: Optional[Outcome] = None
    correct: Optional[bool] = None

    if pred_text is None:
        # API error or no response
        outcome = None
        correct = None
    elif pred_norm is None:
        # Invalid: output exists but could not parse
        outcome = Outcome.INVALID
        correct = None
    elif gold is not None:
        # Valid: parsed to YES/NO
        if pred_norm == gold:
            outcome = Outcome.VALID_CORRECT
            correct = True
        else:
            outcome = Outcome.VALID_INCORRECT
            correct = False
    else:
        # Valid parse but no gold label
        outcome = None
        correct = None

    # Compute group_id for perturbation groups
    group_id = compute_group_id(item)

    return EvalResult(
        item_id=item.get("id", ""),
        batch_file=item.get("_batch_file", ""),
        task_family=task_family,
        bias_family=bias_family,
        dialogue_id=item.get("dialogue_id"),
        turn_index=turn_index,
        system_id=system_id,
        gold=gold,
        pred_raw=pred_text
        if pred_text is not None
        else f"ERROR: {error_msg}"
        if error_msg
        else None,
        pred_norm=pred_norm,
        correct=correct,
        error_type=error_type,
        question=q,
        outcome=outcome,
        group_id=group_id,
    )


def evaluate_items_with_parallelism(
    items: List[Dict[str, Any]],
    client: ModelClient,
    numeric_fact_map: Optional[Dict[str, str]] = None,
    model_name: str = "unknown",
    mode: str = "zeroshot",
    max_workers: Optional[int] = None,
    checkpoint_file: Optional[str] = None,
    checkpoint_interval: int = 50,
    debug: bool = False,
    debug_samples: int = 10,
    output_dir: Optional[str] = None,
    cache: Optional['ResponseCache'] = None,
) -> List[EvalResult]:
    """Evaluate items with parallel execution, rate limiting, and retry logic.

    Args:
        items: List of question items.
        client: Model client.
        numeric_fact_map: System ID to numeric facts mapping.
        model_name: Model name for provider-specific rate limiting.
        mode: Evaluation mode.
        max_workers: Override max workers.
        checkpoint_file: Path to checkpoint file for resume.
        checkpoint_interval: Save checkpoint every N items.
        debug: Enable debug mode.
        debug_samples: Number of debug samples to save.
        output_dir: Output directory for debug samples.
        cache: Optional ResponseCache for caching responses.

    Returns:
        List of EvalResult objects.
    """
    numeric_fact_map = numeric_fact_map or {}

    policy = get_provider_policy(model_name)
    workers = max_workers if max_workers is not None else policy["max_workers"]
    delay = policy["delay"]

    print(f"[CONFIG] Provider policy for '{model_name}': {workers} workers, {delay}s delay")

    results: List[EvalResult] = []
    completed_ids = set()

    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"[RESUME] Loading checkpoint: {checkpoint_file}")
        with open(checkpoint_file, "r") as f:
            checkpoint_data = json.load(f)
            completed_ids = set(checkpoint_data.get("completed_ids", []))
            for r in checkpoint_data.get("results", []):
                results.append(EvalResult(**r))
        print(
            f"[RESUME] Loaded {len(results)} previous results, "
            f"skipping {len(completed_ids)} items"
        )

    remaining_items = [
        item for item in items if item.get("id", "") not in completed_ids
    ]

    if not remaining_items:
        print("[INFO] All items already completed!")
        return results

    total = len(items)
    print(f"[INFO] Processing {len(remaining_items)} items with {workers} parallel workers...")

    debug_results = []
    eval_start_time = time.time()
    items_processed = 0

    if workers == 1:
        print("[INFO] Running in SEQUENTIAL mode (max_workers=1)")
        for idx, item in enumerate(remaining_items, 1):
            result = evaluate_single_item_robust(
                item, client, numeric_fact_map, delay=delay,
                cache=cache, model_name=model_name, mode=mode
            )
            results.append(result)
            items_processed += 1

            if debug and len(debug_results) < debug_samples:
                debug_results.append(
                    {
                        "item_id": result.item_id,
                        "question": item.get("question", ""),
                        "prompt": client.build_prompt(item.get("question", ""), {}),
                        "pred_raw": result.pred_raw,
                        "pred_norm": result.pred_norm,
                        "gold": result.gold,
                        "correct": result.correct,
                    }
                )

            completed = len(results)
            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - eval_start_time
                items_per_sec = items_processed / elapsed if elapsed > 0 else 0
                remaining = len(remaining_items) - items_processed
                eta_seconds = remaining / items_per_sec if items_per_sec > 0 else 0
                eta_min = eta_seconds / 60

                pct = 100 * completed // total
                correct_str = (
                    "OK"
                    if result.correct is True
                    else "FAIL"
                    if result.correct is False
                    else "?"
                )
                print(
                    f"Progress: {completed}/{total} ({pct}%) - "
                    f"Last: {result.pred_norm} (gold: {result.gold}, {correct_str}) | "
                    f"Speed: {items_per_sec:.1f} items/s | ETA: {eta_min:.1f}m"
                )

            if checkpoint_file and completed % checkpoint_interval == 0:
                _save_checkpoint(checkpoint_file, results, completed_ids)
                print(f"[CHECKPOINT] Saved at {completed} items")

    else:
        print(f"[INFO] Running in PARALLEL mode ({workers} workers)")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {
                executor.submit(
                    evaluate_single_item_robust,
                    item,
                    client,
                    numeric_fact_map,
                    delay,
                    cache,
                    model_name,
                    mode,
                ): item
                for item in remaining_items
            }

            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                    items_processed += 1
                    item = future_to_item[future]

                    if debug and len(debug_results) < debug_samples:
                        debug_results.append(
                            {
                                "item_id": result.item_id,
                                "question": item.get("question", ""),
                                "prompt": client.build_prompt(
                                    item.get("question", ""), {}
                                ),
                                "pred_raw": result.pred_raw,
                                "pred_norm": result.pred_norm,
                                "gold": result.gold,
                                "correct": result.correct,
                            }
                        )

                    completed = len(results)
                    if completed % 10 == 0 or completed == total:
                        elapsed = time.time() - eval_start_time
                        items_per_sec = (
                            items_processed / elapsed if elapsed > 0 else 0
                        )
                        remaining = len(remaining_items) - items_processed
                        eta_seconds = (
                            remaining / items_per_sec if items_per_sec > 0 else 0
                        )
                        eta_min = eta_seconds / 60

                        pct = 100 * completed // total
                        correct_str = (
                            "OK"
                            if result.correct is True
                            else "FAIL"
                            if result.correct is False
                            else "?"
                        )
                        print(
                            f"Progress: {completed}/{total} ({pct}%) - "
                            f"Last: {result.pred_norm} (gold: {result.gold}, {correct_str}) | "
                            f"Speed: {items_per_sec:.1f} items/s | ETA: {eta_min:.1f}m"
                        )

                    if checkpoint_file and completed % checkpoint_interval == 0:
                        _save_checkpoint(checkpoint_file, results, completed_ids)
                        print(f"[CHECKPOINT] Saved at {completed} items")

                except Exception as e:
                    print(f"[ERROR] Unexpected error processing item: {e}")
                    continue

    if checkpoint_file:
        _save_checkpoint(checkpoint_file, results, completed_ids)
        print(f"[CHECKPOINT] Final save at {len(results)} items")

    if debug and debug_results and output_dir:
        debug_file = os.path.join(output_dir, "debug_samples.jsonl")
        with open(debug_file, "w") as f:
            for sample in debug_results:
                f.write(json.dumps(sample) + "\n")
        print(f"[DEBUG] Saved {len(debug_results)} debug samples to {debug_file}")

    return results


def evaluate_items(
    items: List[Dict[str, Any]],
    client: ModelClient,
    numeric_fact_map: Optional[Dict[str, str]] = None,
) -> List[EvalResult]:
    """Legacy sequential evaluation function.

    Args:
        items: List of question items.
        client: Model client.
        numeric_fact_map: System ID to numeric facts mapping.

    Returns:
        List of EvalResult objects.
    """
    numeric_fact_map = numeric_fact_map or {}
    results: List[EvalResult] = []
    total = len(items)

    for idx, item in enumerate(items, 1):
        q = item["question"]
        system_id = item.get("system_id")
        ctx: Dict[str, Any] = {}

        if client.config.mode == "tool" and system_id in numeric_fact_map:
            ctx["numeric_facts"] = numeric_fact_map[system_id]

        prompt = client.build_prompt(q, context=ctx)
        pred_text = client.call(prompt)
        pred_norm = normalize_label(pred_text)
        gold_raw = (
            item.get("ground_truth")
            or item.get("gold_label")
            or item.get("answer")
            or item.get("label")
        )
        gold = normalize_label(gold_raw)

        # Compute outcome
        outcome: Optional[Outcome] = None
        correct: Optional[bool] = None
        if pred_norm is None:
            outcome = Outcome.INVALID
            correct = None
        elif gold is not None:
            if pred_norm == gold:
                outcome = Outcome.VALID_CORRECT
                correct = True
            else:
                outcome = Outcome.VALID_INCORRECT
                correct = False

        # Compute group_id
        group_id = compute_group_id(item)

        res = EvalResult(
            item_id=item.get("id", ""),
            batch_file=item.get("_batch_file", ""),
            task_family=item.get("type", "unknown"),
            bias_family=item.get("bias_family"),
            dialogue_id=item.get("dialogue_id"),
            turn_index=item.get("turn") or item.get("turn_index"),
            system_id=system_id,
            gold=gold,
            pred_raw=pred_text,
            pred_norm=pred_norm,
            correct=correct,
            error_type=None,
            question=q,
            outcome=outcome,
            group_id=group_id,
        )
        results.append(res)

        if idx % 10 == 0 or idx == total:
            correct_str = (
                "OK"
                if correct is True
                else "FAIL"
                if correct is False
                else "?"
            )
            print(
                f"Progress: {idx}/{total} ({100*idx//total}%) - "
                f"Last: {pred_norm} (gold: {gold}, {correct_str})"
            )

    return results


def _save_checkpoint(
    checkpoint_file: str,
    results: List[EvalResult],
    completed_ids: set,
):
    """Save checkpoint with results and completed IDs.

    Args:
        checkpoint_file: Path to checkpoint file.
        results: Current results list.
        completed_ids: Previously completed item IDs.
    """
    checkpoint_data = {
        "completed_ids": list(set([r.item_id for r in results]) | completed_ids),
        "results": [asdict(r) for r in results],
        "timestamp": time.time(),
    }

    temp_file = checkpoint_file + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(checkpoint_data, f)
    os.rename(temp_file, checkpoint_file)


def save_run_metadata(
    out_dir: str,
    model_name: str,
    mode: str,
    max_workers: int,
    results: List[EvalResult],
) -> None:
    """Save run metadata for debugging and analysis.

    Args:
        out_dir: Output directory.
        model_name: Model identifier.
        mode: Evaluation mode.
        max_workers: Number of parallel workers used.
        results: Evaluation results.
    """
    os.makedirs(out_dir, exist_ok=True)

    total = len(results)
    valid = len([r for r in results if r.correct is not None])
    unanswered = len([r for r in results if r.correct is None and r.gold is not None])
    no_gold = len([r for r in results if r.gold is None])

    metadata = {
        "model_name": model_name,
        "mode": mode,
        "max_workers": max_workers,
        "num_items_total": total,
        "num_items_evaluated": valid,
        "num_items_unanswered": unanswered,
        "num_items_no_gold": no_gold,
        "timestamp": datetime.now().isoformat(),
    }

    path = os.path.join(out_dir, "run_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved run metadata to {path}")


def save_per_item_results(results: List[EvalResult], out_dir: str) -> None:
    """Save per-item results as JSONL.

    Args:
        results: Evaluation results.
        out_dir: Output directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "per_item_results.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"Saved per-item results to {path}")


def save_summary_json(summary: Dict[str, Any], out_dir: str) -> None:
    """Save summary statistics as JSON.

    Args:
        summary: Summary dict from compute_summary.
        out_dir: Output directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {path}")


def save_csvs(
    summary: Dict[str, Any],
    out_dir: str,
    model_name: str,
    mode: str,
) -> None:
    """Save metrics as CSV files.

    Args:
        summary: Summary dict from compute_summary.
        out_dir: Output directory.
        model_name: Model identifier.
        mode: Evaluation mode.
    """
    os.makedirs(out_dir, exist_ok=True)

    overview_path = os.path.join(out_dir, "metrics_overview.csv")
    header = [
        "model",
        "mode",
        "overall_accuracy",
        "dialogue_accuracy",
        "contradiction_rate",
    ]
    write_header = not os.path.exists(overview_path)
    with open(overview_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(
            [
                model_name,
                mode,
                summary.get("overall_accuracy"),
                summary.get("dialogue_accuracy"),
                summary.get("contradiction_rate"),
            ]
        )
    print(f"Appended overview metrics to {overview_path}")

    task_acc = summary.get("task_accuracy", {})
    task_path = os.path.join(out_dir, "accuracy_by_task.csv")
    with open(task_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_family", "accuracy"])
        for t, acc in sorted(task_acc.items()):
            writer.writerow([t, acc])
    print(f"Saved accuracy-by-task to {task_path}")

    bias_err = summary.get("bias_error", {})
    bias_path = os.path.join(out_dir, "bias_errors.csv")
    with open(bias_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bias_family", "error_rate"])
        for b, err in sorted(bias_err.items()):
            writer.writerow([b, err])
    print(f"Saved bias error rates to {bias_path}")


def save_figures(summary: Dict[str, Any], out_dir: str) -> None:
    """Save metric visualization figures.

    Args:
        summary: Summary dict from compute_summary.
        out_dir: Output directory.
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    task_acc = summary.get("task_accuracy", {})
    if task_acc:
        labels = list(task_acc.keys())
        values = [task_acc[k] for k in labels]

        plt.figure()
        plt.bar(labels, values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Accuracy")
        plt.title("Accuracy by task family")
        plt.tight_layout()
        task_fig_path = os.path.join(out_dir, "task_accuracy_bar.png")
        plt.savefig(task_fig_path, dpi=300)
        plt.close()
        print(f"Saved task accuracy figure to {task_fig_path}")

    bias_err = summary.get("bias_error", {})
    if bias_err:
        labels = list(bias_err.keys())
        values = [bias_err[k] for k in labels]

        plt.figure()
        plt.bar(labels, values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Error rate")
        plt.title("Bias error rates")
        plt.tight_layout()
        bias_fig_path = os.path.join(out_dir, "bias_error_bar.png")
        plt.savefig(bias_fig_path, dpi=300)
        plt.close()
        print(f"Saved bias error figure to {bias_fig_path}")
