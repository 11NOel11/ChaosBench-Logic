"""Evaluation metrics: accuracy, contradiction rate, FOL violations, bias error."""

import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

from chaosbench.logic.axioms import (
    get_fol_rules,
    check_fol_violations,
    load_system_ontology,
)


class Outcome(str, Enum):
    """Three-way evaluation outcome."""
    VALID_CORRECT = "valid_correct"
    VALID_INCORRECT = "valid_incorrect"
    INVALID = "invalid"


YES_SET = {"yes", "true", "y", "t"}
NO_SET = {"no", "false", "n", "f"}


def normalize_label(text: Optional[str]) -> Optional[str]:
    """Normalize a free-form model answer into 'YES' or 'NO'.

    Uses an 8-step parsing cascade to handle diverse model output formats
    including FINAL_ANSWER markers, CoT outputs, revision patterns,
    markdown formatting, and fallback token matching.

    Args:
        text: Raw model output text.

    Returns:
        "YES", "NO", or None if unparseable.
    """
    if text is None:
        return None

    raw = text.strip()
    if not raw:
        return None

    text_cleaned = re.sub(r"[*_`]", "", raw)

    final_answer_match = re.search(
        r"FINAL[_\s-]*ANSWER\s*[:=]\s*([^\n.,;]+)",
        text_cleaned,
        re.IGNORECASE,
    )
    if final_answer_match:
        answer_part = final_answer_match.group(1).strip()
    else:
        answer_patterns = [
            r"(?:final|ultimate|my)\s+answer\s*[:=]?\s*([^\n.,;]+)",
            r"(?:the\s+)?answer\s+is\s+(.+?)(?:\.|\n|$)",
            r"(?:i\s+)?answer\s*[:=]\s*([^\n.,;]+)",
            r"therefore\s*[,:=]?\s*([^\n.,;]+)",
            r"conclusion\s*[:=]\s*([^\n.,;]+)",
            r"so\s+(?:the\s+answer\s+is\s+)?([^\n.,;]+)",
        ]

        answer_part = None
        for pattern in answer_patterns:
            match = re.search(pattern, text_cleaned, re.IGNORECASE)
            if match:
                answer_part = match.group(1).strip()
                break

        if answer_part is None:
            lines = [line.strip() for line in text_cleaned.split("\n") if line.strip()]
            if lines:
                answer_part = lines[-1]

        if not answer_part:
            answer_part = text_cleaned.strip()

    if not answer_part:
        return None

    cleaned = re.sub(r"[^\w\s]", " ", answer_part)
    cleaned = " ".join(cleaned.split()).lower()

    if not cleaned:
        full_lower = raw.lower()
        if re.search(r"\byes\b", full_lower):
            return "YES"
        if re.search(r"\bno\b", full_lower):
            return "NO"
        if re.search(r"\btrue\b", full_lower):
            return "YES"
        if re.search(r"\bfalse\b", full_lower):
            return "NO"
        return None

    tokens = cleaned.split()

    if not tokens:
        return None

    last_yes_token = None
    last_no_token = None
    for i, token in enumerate(tokens):
        if token in {"yes", "true"}:
            last_yes_token = i
        if token in {"no", "false"}:
            last_no_token = i

    if last_yes_token is not None and last_no_token is not None:
        if last_yes_token > last_no_token:
            return "YES"
        else:
            return "NO"
    elif last_yes_token is not None:
        return "YES"
    elif last_no_token is not None:
        return "NO"

    full_lower = raw.lower()
    last_yes_idx = max(
        full_lower.rfind(" yes"),
        full_lower.rfind(" yes."),
        full_lower.rfind(" yes,"),
        full_lower.rfind("\nyes"),
        full_lower.rfind(" true"),
        full_lower.rfind(" true."),
    )
    last_no_idx = max(
        full_lower.rfind(" no"),
        full_lower.rfind(" no."),
        full_lower.rfind(" no,"),
        full_lower.rfind("\nno"),
        full_lower.rfind(" false"),
        full_lower.rfind(" false."),
    )

    if last_yes_idx > last_no_idx and last_yes_idx >= 0:
        return "YES"
    elif last_no_idx > last_yes_idx and last_no_idx >= 0:
        return "NO"

    return None


@dataclass
class EvalResult:
    """Result of evaluating a single benchmark item.

    Attributes:
        item_id: Unique item identifier.
        batch_file: Source batch file name.
        task_family: Task category (e.g. "atomic", "multi_hop").
        bias_family: Optional bias category.
        dialogue_id: Optional dialogue identifier for multi-turn items.
        turn_index: Optional turn position within dialogue.
        system_id: Optional dynamical system identifier.
        gold: Ground truth label ("YES"/"NO").
        pred_raw: Raw model output text.
        pred_norm: Normalized prediction ("YES"/"NO"/None).
        correct: Whether prediction matches gold.
        error_type: Error classification if API call failed.
        question: Original question text.
        outcome: Three-way outcome (VALID_CORRECT/VALID_INCORRECT/INVALID).
        group_id: Optional perturbation group identifier for flip rate analysis.
    """

    item_id: str
    batch_file: str
    task_family: str
    bias_family: Optional[str]
    dialogue_id: Optional[str]
    turn_index: Optional[int]
    system_id: Optional[str]
    gold: Optional[str]
    pred_raw: Optional[str]
    pred_norm: Optional[str]
    correct: Optional[bool]
    error_type: Optional[str] = None
    question: Optional[str] = None
    outcome: Optional[Outcome] = None
    group_id: Optional[str] = None


def compute_summary(results: List[EvalResult]) -> Dict[str, Any]:
    """Compute summary statistics from evaluation results.

    Includes coverage-aware metrics:
    - coverage: fraction of items with valid (parseable) output
    - accuracy_valid: accuracy on valid items only
    - effective_accuracy: coverage * accuracy_valid
    - invalid_rate: fraction of unparseable outputs

    Args:
        results: List of EvalResult objects.

    Returns:
        Dict with keys: overall_accuracy, coverage, accuracy_valid,
        effective_accuracy, invalid_rate, task_accuracy, dialogue_accuracy,
        contradiction_rate, bias_error, avg_violations_per_dialogue,
        violations_breakdown, flip_rate, and optionally error_breakdown.
    """
    summary: Dict[str, Any] = {}

    total_items = len(results)
    valid = [r for r in results if r.correct is not None]
    invalid = [r for r in results if r.correct is None and r.pred_raw is not None]
    unanswered = [r for r in results if r.pred_raw is None or (r.error_type is not None)]
    no_gold = [r for r in results if r.gold is None]

    print(f"\n[SUMMARY] Result breakdown:")
    print(f"  Total items: {total_items}")
    print(f"  Valid (parsed to TRUE/FALSE): {len(valid)}")
    print(f"  Invalid (failed to parse): {len(invalid)}")
    print(f"  Unanswered (API errors): {len(unanswered)}")
    print(f"  No gold label: {len(no_gold)}")

    # Coverage-aware metrics
    items_with_output = [r for r in results if r.pred_raw is not None and r.gold is not None]
    if items_with_output:
        coverage = len(valid) / len(items_with_output)
        invalid_rate = len(invalid) / len(items_with_output)
        summary["coverage"] = coverage
        summary["invalid_rate"] = invalid_rate
        print(f"\n[COVERAGE] Coverage: {coverage*100:.1f}% | Invalid rate: {invalid_rate*100:.1f}%")
    else:
        summary["coverage"] = None
        summary["invalid_rate"] = None

    if unanswered:
        error_counts: Dict[str, int] = defaultdict(int)
        for r in unanswered:
            if r.error_type:
                error_counts[r.error_type] += 1
            else:
                error_counts["Unknown"] += 1

        print(f"\n[ERROR BREAKDOWN] Unanswered items by error type:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")
        summary["error_breakdown"] = dict(error_counts)

    if valid:
        num_correct = sum(1 for r in valid if r.correct is True)
        accuracy_valid = num_correct / len(valid)
        summary["accuracy_valid"] = accuracy_valid
        summary["overall_accuracy"] = accuracy_valid  # Keep for backward compatibility

        # Effective accuracy = coverage * accuracy_valid
        if summary["coverage"] is not None:
            summary["effective_accuracy"] = summary["coverage"] * accuracy_valid
        else:
            summary["effective_accuracy"] = None

        print(f"\n[ACCURACY] Accuracy (on valid): {accuracy_valid*100:.1f}% ({num_correct}/{len(valid)})")
        if summary["effective_accuracy"] is not None:
            print(f"           Effective accuracy (coverage × accuracy): {summary['effective_accuracy']*100:.1f}%")

        if summary["overall_accuracy"] == 0.0 and len(valid) > 50:
            print("\nWARNING: Overall accuracy is 0.0% with >50 items!")
            print("  Possible issues:")
            print("    - Check normalization (see debug_samples.jsonl)")
            print("    - Many items may have failed due to rate limits")
            print("    - Model responses may not match expected format")
            print(f"    - {len(unanswered)} items had all retries fail\n")
    else:
        summary["accuracy_valid"] = None
        summary["overall_accuracy"] = None
        summary["effective_accuracy"] = None
        if total_items > 0:
            print("\nWARNING: No valid items to compute accuracy!")

    # Compute flip rate for perturbation groups
    flip_rate = _compute_flip_rate(results)
    if flip_rate is not None:
        summary["flip_rate"] = flip_rate
        print(f"\n[FLIP RATE] Perturbation groups: {flip_rate*100:.1f}% had disagreements")
    else:
        summary["flip_rate"] = None

    # Compute imbalance-robust metrics
    balanced_acc = compute_balanced_accuracy(results)
    mcc = compute_mcc(results)
    per_class = compute_per_class_metrics(results)

    if balanced_acc is not None:
        summary["balanced_accuracy"] = balanced_acc
        print(f"\n[BALANCED METRICS] Balanced accuracy: {balanced_acc*100:.1f}%")
    else:
        summary["balanced_accuracy"] = None

    if mcc is not None:
        summary["mcc"] = mcc
        print(f"                   MCC: {mcc:.3f}")
    else:
        summary["mcc"] = None

    if per_class:
        summary["per_class_metrics"] = per_class
        print(f"                   YES - P: {per_class['YES']['precision']:.3f}, R: {per_class['YES']['recall']:.3f}, F1: {per_class['YES']['f1']:.3f}")
        print(f"                   NO  - P: {per_class['NO']['precision']:.3f}, R: {per_class['NO']['recall']:.3f}, F1: {per_class['NO']['f1']:.3f}")
    else:
        summary["per_class_metrics"] = {}

    by_task: Dict[str, List[EvalResult]] = defaultdict(list)
    for r in valid:
        by_task[r.task_family].append(r)

    task_acc: Dict[str, float] = {}
    for t, lst in by_task.items():
        n = len(lst)
        c = sum(1 for rr in lst if rr.correct is True)
        task_acc[t] = c / n if n > 0 else 0.0
    summary["task_accuracy"] = task_acc

    bias_items = [r for r in valid if r.bias_family is not None]

    if not bias_items and total_items > 0:
        print(
            "\nWARNING: No items contained a bias_family label; bias_error will be empty."
        )

    by_bias: Dict[str, List[EvalResult]] = defaultdict(list)
    for r in bias_items:
        assert r.bias_family is not None
        by_bias[r.bias_family].append(r)

    bias_err: Dict[str, float] = {}
    for b, lst in by_bias.items():
        n = len(lst)
        c = sum(1 for rr in lst if rr.correct is True)
        acc = c / n if n > 0 else 0.0
        bias_err[b] = 1.0 - acc
    summary["bias_error"] = bias_err

    dialogues: Dict[str, List[EvalResult]] = defaultdict(list)
    for r in results:
        if r.dialogue_id is not None:
            dialogues[r.dialogue_id].append(r)

    dialogue_accs: List[float] = []
    contradiction_count = 0

    for did, turns in dialogues.items():
        turns_sorted = sorted(
            [t for t in turns if t.turn_index is not None],
            key=lambda x: x.turn_index if x.turn_index is not None else 0,
        )
        if not turns_sorted:
            continue

        all_known = all(t.correct is not None for t in turns_sorted)
        all_correct = all(
            t.correct is True for t in turns_sorted if t.correct is not None
        )
        if all_known and all_correct:
            dialogue_accs.append(1.0)
        else:
            dialogue_accs.append(0.0)

        answers_by_key: Dict[tuple, set] = defaultdict(set)
        for t in turns_sorted:
            key = (t.system_id, t.task_family)
            if t.pred_norm is not None:
                answers_by_key[key].add(t.pred_norm)

        for ans_set in answers_by_key.values():
            if "YES" in ans_set and "NO" in ans_set:
                contradiction_count += 1
                break

    if dialogue_accs:
        summary["dialogue_accuracy"] = sum(dialogue_accs) / len(dialogue_accs)
    else:
        summary["dialogue_accuracy"] = None

    if dialogues:
        summary["contradiction_rate"] = contradiction_count / len(dialogues)
    else:
        summary["contradiction_rate"] = None

    ontology = load_system_ontology(systems_dir="systems")

    all_dialogue_groups: Dict[str, List[EvalResult]] = {}

    for did, turns in dialogues.items():
        all_dialogue_groups[did] = turns

    single_questions = [r for r in results if r.dialogue_id is None]
    for r in single_questions:
        synthetic_id = f"single_{r.item_id}"
        all_dialogue_groups[synthetic_id] = [r]

    from chaosbench.logic.extract import extract_predicate_from_question

    violation_counts: List[int] = []

    for dialogue_id, turns in all_dialogue_groups.items():
        predictions_by_system: Dict[str, Dict[str, str]] = defaultdict(dict)

        for turn in turns:
            if turn.system_id and turn.pred_norm and turn.question:
                predicate = extract_predicate_from_question(turn.question)
                if predicate:
                    predictions_by_system[turn.system_id][predicate] = turn.pred_norm

        num_violations = 0
        for system_id, predictions in predictions_by_system.items():
            violations = check_fol_violations(predictions)
            num_violations += len(violations)

        violation_counts.append(num_violations)

    if violation_counts:
        summary["avg_violations_per_dialogue"] = sum(violation_counts) / len(
            violation_counts
        )

        violations_breakdown: Dict[str, int] = {
            "0_violations": 0,
            "1_violation": 0,
            "2_violations": 0,
            "3+_violations": 0,
        }
        for count in violation_counts:
            if count == 0:
                violations_breakdown["0_violations"] += 1
            elif count == 1:
                violations_breakdown["1_violation"] += 1
            elif count == 2:
                violations_breakdown["2_violations"] += 1
            else:
                violations_breakdown["3+_violations"] += 1

        summary["violations_breakdown"] = violations_breakdown

        print(f"\n[FOL VIOLATIONS] Dialogue violation statistics:")
        print(
            f"  Avg violations per dialogue: {summary['avg_violations_per_dialogue']:.2f}"
        )
        print(f"  Breakdown: {dict(violations_breakdown)}")
    else:
        summary["avg_violations_per_dialogue"] = None
        summary["violations_breakdown"] = {}

    return summary


def _compute_flip_rate(results: List[EvalResult]) -> Optional[float]:
    """Compute flip rate across perturbation groups.

    Flip rate = fraction of groups where answers disagree across variants.

    Args:
        results: List of EvalResult objects.

    Returns:
        Fraction of groups with disagreements, or None if no groups.
    """
    # Group by group_id
    by_group: Dict[str, List[EvalResult]] = defaultdict(list)
    for r in results:
        if r.group_id is not None and r.pred_norm is not None:
            by_group[r.group_id].append(r)

    if not by_group:
        return None

    flipped = 0
    for group_id, group_results in by_group.items():
        # Check if group has both YES and NO predictions
        predictions = {r.pred_norm for r in group_results if r.pred_norm is not None}
        if "YES" in predictions and "NO" in predictions:
            flipped += 1

    return flipped / len(by_group)


def compute_balanced_accuracy(results: List[EvalResult]) -> Optional[float]:
    """Compute balanced accuracy (average of per-class recall).

    Robust to class imbalance. Balanced accuracy = (TPR + TNR) / 2.

    Args:
        results: List of EvalResult objects.

    Returns:
        Balanced accuracy float, or None if insufficient data.
    """
    valid = [r for r in results if r.correct is not None and r.gold is not None]
    if not valid:
        return None

    # Count by class
    pos_correct = sum(1 for r in valid if r.gold == "YES" and r.correct is True)
    pos_total = sum(1 for r in valid if r.gold == "YES")
    neg_correct = sum(1 for r in valid if r.gold == "NO" and r.correct is True)
    neg_total = sum(1 for r in valid if r.gold == "NO")

    if pos_total == 0 or neg_total == 0:
        # Only one class present
        return None

    tpr = pos_correct / pos_total  # True positive rate (recall for YES)
    tnr = neg_correct / neg_total  # True negative rate (recall for NO)

    return (tpr + tnr) / 2.0


def compute_mcc(results: List[EvalResult]) -> Optional[float]:
    """Compute Matthews correlation coefficient (MCC).

    MCC is robust to class imbalance and ranges from -1 to +1.

    Args:
        results: List of EvalResult objects.

    Returns:
        MCC float, or None if insufficient data.
    """
    valid = [r for r in results if r.correct is not None and r.gold is not None]
    if not valid:
        return None

    # Confusion matrix counts
    tp = sum(1 for r in valid if r.gold == "YES" and r.pred_norm == "YES")
    tn = sum(1 for r in valid if r.gold == "NO" and r.pred_norm == "NO")
    fp = sum(1 for r in valid if r.gold == "NO" and r.pred_norm == "YES")
    fn = sum(1 for r in valid if r.gold == "YES" and r.pred_norm == "NO")

    # MCC formula
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    if denominator == 0:
        return 0.0

    return numerator / denominator


def compute_per_class_metrics(results: List[EvalResult]) -> Dict[str, Any]:
    """Compute per-class precision, recall, and F1.

    Args:
        results: List of EvalResult objects.

    Returns:
        Dict with 'YES' and 'NO' class metrics.
    """
    valid = [r for r in results if r.correct is not None and r.gold is not None]
    if not valid:
        return {}

    # Counts for YES class
    tp_yes = sum(1 for r in valid if r.gold == "YES" and r.pred_norm == "YES")
    fp_yes = sum(1 for r in valid if r.gold == "NO" and r.pred_norm == "YES")
    fn_yes = sum(1 for r in valid if r.gold == "YES" and r.pred_norm == "NO")

    # Counts for NO class
    tp_no = sum(1 for r in valid if r.gold == "NO" and r.pred_norm == "NO")
    fp_no = sum(1 for r in valid if r.gold == "YES" and r.pred_norm == "NO")
    fn_no = sum(1 for r in valid if r.gold == "NO" and r.pred_norm == "YES")

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    # YES metrics
    precision_yes = safe_div(tp_yes, tp_yes + fp_yes)
    recall_yes = safe_div(tp_yes, tp_yes + fn_yes)
    f1_yes = safe_div(2 * precision_yes * recall_yes, precision_yes + recall_yes)

    # NO metrics
    precision_no = safe_div(tp_no, tp_no + fp_no)
    recall_no = safe_div(tp_no, tp_no + fn_no)
    f1_no = safe_div(2 * precision_no * recall_no, precision_no + recall_no)

    return {
        "YES": {
            "precision": precision_yes,
            "recall": recall_yes,
            "f1": f1_yes,
            "support": tp_yes + fn_yes,
        },
        "NO": {
            "precision": precision_no,
            "recall": recall_no,
            "f1": f1_no,
            "support": tp_no + fn_no,
        },
    }


def compute_per_task_accuracy(
    results: List[EvalResult],
) -> Dict[str, float]:
    """Compute accuracy for each task family.

    Args:
        results: List of EvalResult objects.

    Returns:
        Dict mapping task_family to accuracy float.
    """
    by_task: Dict[str, List[EvalResult]] = defaultdict(list)
    valid = [r for r in results if r.correct is not None]
    for r in valid:
        by_task[r.task_family].append(r)

    task_acc: Dict[str, float] = {}
    for t, lst in by_task.items():
        n = len(lst)
        c = sum(1 for rr in lst if rr.correct is True)
        task_acc[t] = c / n if n > 0 else 0.0
    return task_acc


def compute_contradiction_rate(results: List[EvalResult]) -> Optional[float]:
    """Compute contradiction rate across dialogues.

    Args:
        results: List of EvalResult objects.

    Returns:
        Fraction of dialogues containing contradictions, or None if no dialogues.
    """
    dialogues: Dict[str, List[EvalResult]] = defaultdict(list)
    for r in results:
        if r.dialogue_id is not None:
            dialogues[r.dialogue_id].append(r)

    if not dialogues:
        return None

    contradiction_count = 0
    for did, turns in dialogues.items():
        answers_by_key: Dict[tuple, set] = defaultdict(set)
        for t in turns:
            key = (t.system_id, t.task_family)
            if t.pred_norm is not None:
                answers_by_key[key].add(t.pred_norm)

        for ans_set in answers_by_key.values():
            if "YES" in ans_set and "NO" in ans_set:
                contradiction_count += 1
                break

    return contradiction_count / len(dialogues)


def compute_axiom_violation_rate(
    results: List[EvalResult],
) -> Optional[float]:
    """Compute average FOL axiom violations per dialogue.

    Args:
        results: List of EvalResult objects.

    Returns:
        Average number of violations per dialogue group, or None.
    """
    from chaosbench.logic.extract import extract_predicate_from_question

    dialogues: Dict[str, List[EvalResult]] = defaultdict(list)
    for r in results:
        if r.dialogue_id is not None:
            dialogues[r.dialogue_id].append(r)

    single_questions = [r for r in results if r.dialogue_id is None]
    all_groups: Dict[str, List[EvalResult]] = dict(dialogues)
    for r in single_questions:
        all_groups[f"single_{r.item_id}"] = [r]

    if not all_groups:
        return None

    total_violations = 0
    for group_id, turns in all_groups.items():
        predictions_by_system: Dict[str, Dict[str, str]] = defaultdict(dict)
        for turn in turns:
            if turn.system_id and turn.pred_norm and turn.question:
                predicate = extract_predicate_from_question(turn.question)
                if predicate:
                    predictions_by_system[turn.system_id][predicate] = turn.pred_norm

        for system_id, predictions in predictions_by_system.items():
            violations = check_fol_violations(predictions)
            total_violations += len(violations)

    return total_violations / len(all_groups)


@dataclass
class AxisMetricResult:
    """Result for a single metric axis value.

    Attributes:
        axis: Name of the axis dimension.
        value: Specific value for this axis.
        accuracy: Accuracy for this axis value.
        n_correct: Number of correct predictions.
        n_total: Total number of predictions.
        fol_violation_rate: Rate of FOL violations for this axis value.
    """
    axis: str
    value: str
    accuracy: float
    n_correct: int
    n_total: int
    fol_violation_rate: float = 0.0


def compute_axis_metrics(
    results: List[EvalResult],
    axes: Optional[List[str]] = None,
) -> Dict[str, List[AxisMetricResult]]:
    """Compute metrics stratified by multiple axes.

    Args:
        results: List of evaluation results.
        axes: Axes to stratify by. Defaults to ["split", "task_family", "system_category"].
            Supported axes: "split" (from batch_file), "task_family",
            "system_category" (from system_id prefix), "batch_file".

    Returns:
        Dict mapping axis name to list of AxisMetricResult per value.
    """
    from chaosbench.data.splits import SPLIT_ASSIGNMENTS
    from chaosbench.logic.extract import extract_predicate_from_question

    if axes is None:
        axes = ["split", "task_family", "system_category"]

    axis_results: Dict[str, List[AxisMetricResult]] = {}

    for axis_name in axes:
        groups: Dict[str, List[EvalResult]] = defaultdict(list)

        for result in results:
            axis_value = None

            if axis_name == "split":
                batch_name = result.batch_file.replace(".jsonl", "")
                axis_value = SPLIT_ASSIGNMENTS.get(batch_name, "unknown")
            elif axis_name == "task_family":
                axis_value = result.task_family
            elif axis_name == "system_category":
                if result.system_id:
                    if result.system_id.startswith("dysts_"):
                        axis_value = "dysts"
                    else:
                        axis_value = "core"
                else:
                    axis_value = "unknown"
            elif axis_name == "batch_file":
                axis_value = result.batch_file

            if axis_value:
                groups[axis_value].append(result)

        axis_metric_list: List[AxisMetricResult] = []

        for value, group_results in sorted(groups.items()):
            valid_results = [r for r in group_results if r.correct is not None]

            if not valid_results:
                continue

            n_total = len(valid_results)
            n_correct = sum(1 for r in valid_results if r.correct is True)
            accuracy = n_correct / n_total if n_total > 0 else 0.0

            dialogues: Dict[str, List[EvalResult]] = defaultdict(list)
            for r in group_results:
                if r.dialogue_id is not None:
                    dialogues[r.dialogue_id].append(r)

            single_questions = [r for r in group_results if r.dialogue_id is None]
            all_groups: Dict[str, List[EvalResult]] = dict(dialogues)
            for r in single_questions:
                all_groups[f"single_{r.item_id}"] = [r]

            total_violations = 0
            for group_id, turns in all_groups.items():
                predictions_by_system: Dict[str, Dict[str, str]] = defaultdict(dict)
                for turn in turns:
                    if turn.system_id and turn.pred_norm and turn.question:
                        predicate = extract_predicate_from_question(turn.question)
                        if predicate:
                            predictions_by_system[turn.system_id][predicate] = turn.pred_norm

                for system_id, predictions in predictions_by_system.items():
                    violations = check_fol_violations(predictions)
                    total_violations += len(violations)

            fol_violation_rate = total_violations / len(all_groups) if all_groups else 0.0

            axis_metric_list.append(
                AxisMetricResult(
                    axis=axis_name,
                    value=value,
                    accuracy=accuracy,
                    n_correct=n_correct,
                    n_total=n_total,
                    fol_violation_rate=fol_violation_rate,
                )
            )

        axis_results[axis_name] = axis_metric_list

    return axis_results


def format_axis_report(axis_metrics: Dict[str, List[AxisMetricResult]]) -> str:
    """Format axis metrics as a markdown table string.

    Args:
        axis_metrics: Dict mapping axis name to list of AxisMetricResult.

    Returns:
        Markdown-formatted string with tables for each axis.
    """
    lines = []

    for axis_name, metrics in sorted(axis_metrics.items()):
        lines.append(f"\n## {axis_name.replace('_', ' ').title()}\n")
        lines.append("| Value | Accuracy | Correct | Total | FOL Violation Rate |")
        lines.append("|-------|----------|---------|-------|-------------------|")

        for metric in metrics:
            acc_pct = f"{metric.accuracy * 100:.1f}%"
            fol_rate = f"{metric.fol_violation_rate:.3f}"
            lines.append(
                f"| {metric.value} | {acc_pct} | {metric.n_correct} | {metric.n_total} | {fol_rate} |"
            )

    return "\n".join(lines)
