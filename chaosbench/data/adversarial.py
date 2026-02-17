"""Adversarial question generation techniques for ChaosBench-Logic v2."""

import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from chaosbench.data.schemas import Question
from chaosbench.data.bifurcations import (
    BIFURCATION_DATA,
    get_regime_at_param,
    is_chaotic_regime,
)
from chaosbench.logic.ontology import PREDICATES

CONFUSABLE_PAIRS = [
    ("Sensitive", "PointUnpredictable"),
    ("Chaotic", "Random"),
    ("StatPredictable", "Deterministic"),
    ("QuasiPeriodic", "Periodic"),
    ("StrangeAttr", "Chaotic"),
    ("FixedPointAttr", "Periodic"),
]

PREMISE_TEMPLATES = [
    "Given that the system is {pred_lower}, ",
    "Knowing that this system exhibits {pred_lower} behavior, ",
    "Considering that {pred_lower} has been established for this system, ",
    "Since the system is known to be {pred_lower}, ",
]

ATOMIC_TEMPLATES = [
    "Is the {system_name} system {pred_lower}?",
    "Does the {system_name} system exhibit {pred_lower} behavior?",
    "Can the {system_name} system be characterized as {pred_lower}?",
]

NEAR_MISS_TEMPLATES = [
    "At {param_name}={param_value}, is the {system_name} system chaotic?",
    "Does the {system_name} system exhibit chaos when {param_name}={param_value}?",
    "Is the {system_name} system chaotic for the parameter setting {param_name}={param_value}?",
]

PREDICATE_DISPLAY = {
    "Chaotic": "chaotic",
    "Deterministic": "deterministic",
    "PosLyap": "positive Lyapunov exponent",
    "Sensitive": "sensitive to initial conditions",
    "StrangeAttr": "strange attractor",
    "PointUnpredictable": "pointwise unpredictable",
    "StatPredictable": "statistically predictable",
    "QuasiPeriodic": "quasi-periodic",
    "Random": "random",
    "FixedPointAttr": "fixed point attractor",
    "Periodic": "periodic",
}


def generate_misleading_premise(
    question: Question,
    system_truth: Dict[str, bool],
    seed: int = 42,
) -> Question:
    """Add a true-but-irrelevant fact that might mislead the model.

    Picks a true predicate from the system truth assignment that is not
    already tested by the question, then prepends it as a premise. The
    ground truth does not change.

    Args:
        question: Original question to augment.
        system_truth: Ground truth predicate assignment for the system.
        seed: Random seed for reproducibility.

    Returns:
        Modified Question with misleading premise prepended.
    """
    rng = random.Random(seed)

    tested = set(question.predicates)
    true_preds = [p for p, v in system_truth.items() if v and p not in tested]

    if not true_preds:
        true_preds = [p for p, v in system_truth.items() if v]

    if not true_preds:
        return copy.deepcopy(question)

    chosen = rng.choice(true_preds)
    pred_lower = PREDICATE_DISPLAY.get(chosen, chosen.lower())
    template = rng.choice(PREMISE_TEMPLATES)
    prefix = template.format(pred_lower=pred_lower)

    modified = copy.deepcopy(question)
    original_first = modified.question_text[0].lower()
    modified.question_text = prefix + original_first + modified.question_text[1:]
    modified.item_id = f"{question.item_id}_mislead"
    modified.task_family = "adversarial_misleading"
    modified.metadata = dict(modified.metadata)
    modified.metadata["adversarial_type"] = "misleading_premise"
    modified.metadata["misleading_predicate"] = chosen

    return modified


def generate_near_miss(
    system_id: str,
    system_name: str,
    system_truth: Dict[str, bool],
    bifurcation_data: Dict,
    seed: int = 42,
) -> Question:
    """Ask about a system at a nearby parameter value where the regime differs.

    Uses bifurcation data to find parameter values near regime transitions.
    The generated question asks whether the system is chaotic at a parameter
    value just across a transition boundary from the current regime.

    Args:
        system_id: System identifier (e.g. "logistic", "lorenz63").
        system_name: Human-readable system name.
        system_truth: Ground truth predicate assignment for the system.
        bifurcation_data: Dict mapping system_id to BifurcationInfo objects.
        seed: Random seed for reproducibility.

    Returns:
        Question about the system at a near-miss parameter value.
    """
    rng = random.Random(seed)

    if system_id not in bifurcation_data:
        ground_truth = "YES" if system_truth.get("Chaotic", False) else "NO"
        return Question(
            item_id=f"{system_id}_nearmiss_{seed}",
            question_text=f"Is the {system_name} system chaotic?",
            system_id=system_id,
            task_family="adversarial_nearmiss",
            ground_truth=ground_truth,
            predicates=["Chaotic"],
            metadata={"adversarial_type": "near_miss", "fallback": True},
        )

    info = bifurcation_data[system_id]
    param_name = info.parameter_name
    transitions = info.transitions

    transition_pairs = []
    for i in range(len(transitions) - 1):
        regime_a = transitions[i].regime
        regime_b = transitions[i + 1].regime
        chaotic_a = is_chaotic_regime(regime_a)
        chaotic_b = is_chaotic_regime(regime_b)
        if chaotic_a != chaotic_b:
            boundary = transitions[i + 1].param_value
            transition_pairs.append((boundary, chaotic_a, chaotic_b))

    if not transition_pairs:
        ground_truth = "YES" if system_truth.get("Chaotic", False) else "NO"
        return Question(
            item_id=f"{system_id}_nearmiss_{seed}",
            question_text=f"Is the {system_name} system chaotic?",
            system_id=system_id,
            task_family="adversarial_nearmiss",
            ground_truth=ground_truth,
            predicates=["Chaotic"],
            metadata={"adversarial_type": "near_miss", "no_boundary": True},
        )

    boundary, chaotic_below, chaotic_above = rng.choice(transition_pairs)

    epsilon = boundary * 0.02 if boundary != 0 else 0.01
    if rng.random() < 0.5:
        param_value = round(boundary - epsilon, 4)
        is_chaotic = chaotic_below
    else:
        param_value = round(boundary + epsilon, 4)
        is_chaotic = chaotic_above

    ground_truth = "YES" if is_chaotic else "NO"
    template = rng.choice(NEAR_MISS_TEMPLATES)
    question_text = template.format(
        param_name=param_name,
        param_value=param_value,
        system_name=system_name,
    )

    return Question(
        item_id=f"{system_id}_nearmiss_{seed}",
        question_text=question_text,
        system_id=system_id,
        task_family="adversarial_nearmiss",
        ground_truth=ground_truth,
        predicates=["Chaotic"],
        metadata={
            "adversarial_type": "near_miss",
            "boundary_param": boundary,
            "query_param": param_value,
            "param_name": param_name,
        },
    )


def generate_predicate_confusion(
    question: Question,
    seed: int = 42,
) -> Question:
    """Mix similar predicates that models commonly confuse.

    Replaces the target predicate in the question with a confusable pair
    member and adjusts the ground truth accordingly. If the question tests
    predicate A and (A, B) is a confusable pair, the question text is
    rewritten to ask about B instead.

    Args:
        question: Original question targeting a specific predicate.
        seed: Random seed for reproducibility.

    Returns:
        Modified Question with a confusable predicate substituted.
    """
    rng = random.Random(seed)

    if not question.predicates:
        return copy.deepcopy(question)

    target_pred = question.predicates[0]

    swap_candidates = []
    for a, b in CONFUSABLE_PAIRS:
        if a == target_pred:
            swap_candidates.append(b)
        elif b == target_pred:
            swap_candidates.append(a)

    if not swap_candidates:
        return copy.deepcopy(question)

    new_pred = rng.choice(swap_candidates)

    old_display = PREDICATE_DISPLAY.get(target_pred, target_pred.lower())
    new_display = PREDICATE_DISPLAY.get(new_pred, new_pred.lower())

    modified = copy.deepcopy(question)
    modified.question_text = modified.question_text.replace(old_display, new_display)

    if modified.question_text == question.question_text:
        modified.question_text = modified.question_text.replace(
            target_pred.lower(), new_display
        )

    modified.item_id = f"{question.item_id}_confuse"
    modified.task_family = "adversarial_confusion"
    modified.predicates = [new_pred]
    modified.ground_truth = "UNKNOWN"
    modified.metadata = dict(modified.metadata)
    modified.metadata["adversarial_type"] = "predicate_confusion"
    modified.metadata["original_predicate"] = target_pred
    modified.metadata["swapped_predicate"] = new_pred

    return modified


def generate_adversarial_set(
    systems: Dict[str, Dict],
    n_per_type: int = 5,
    seed: int = 42,
) -> List[Question]:
    """Generate a complete adversarial question set.

    Creates n_per_type questions of each adversarial type (misleading
    premise, near miss, predicate confusion) across the provided systems.

    Args:
        systems: Dict mapping system_id to system info dicts. Each info dict
            must contain "name" (str) and "truth" (Dict[str, bool]).
        n_per_type: Number of questions to generate per adversarial type.
        seed: Random seed for reproducibility.

    Returns:
        List of adversarial Question objects.
    """
    rng = random.Random(seed)
    results: List[Question] = []
    system_ids = sorted(systems.keys())

    for i in range(n_per_type):
        sid = system_ids[i % len(system_ids)]
        info = systems[sid]
        truth = info["truth"]
        name = info["name"]

        pred = rng.choice(PREDICATES)
        pred_display = PREDICATE_DISPLAY.get(pred, pred.lower())
        gt = "YES" if truth.get(pred, False) else "NO"
        template = rng.choice(ATOMIC_TEMPLATES)
        base_q = Question(
            item_id=f"adv_mislead_{i}",
            question_text=template.format(system_name=name, pred_lower=pred_display),
            system_id=sid,
            task_family="atomic",
            ground_truth=gt,
            predicates=[pred],
        )
        results.append(generate_misleading_premise(base_q, truth, seed=seed + i))

    for i in range(n_per_type):
        sid = system_ids[i % len(system_ids)]
        info = systems[sid]
        truth = info["truth"]
        name = info["name"]
        results.append(
            generate_near_miss(sid, name, truth, BIFURCATION_DATA, seed=seed + i)
        )

    for i in range(n_per_type):
        sid = system_ids[i % len(system_ids)]
        info = systems[sid]
        truth = info["truth"]
        name = info["name"]

        pred = rng.choice(PREDICATES)
        pred_display = PREDICATE_DISPLAY.get(pred, pred.lower())
        gt = "YES" if truth.get(pred, False) else "NO"
        template = rng.choice(ATOMIC_TEMPLATES)
        base_q = Question(
            item_id=f"adv_confuse_{i}",
            question_text=template.format(system_name=name, pred_lower=pred_display),
            system_id=sid,
            task_family="atomic",
            ground_truth=gt,
            predicates=[pred],
        )
        results.append(generate_predicate_confusion(base_q, seed=seed + i))

    return results
