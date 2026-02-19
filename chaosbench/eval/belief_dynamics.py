"""Belief dynamics: divergence curves, instability scores, sensitivity profiles."""

from typing import Dict, List, Optional, Tuple


def hamming_distance(
    b1: Dict[str, str],
    b2: Dict[str, str],
) -> int:
    """Compute Hamming distance between two belief vectors.

    Only counts predicates present in both vectors. Mismatched values
    (including UNKNOWN vs YES/NO) count as distance 1.

    Args:
        b1: First belief vector {predicate: "YES"/"NO"/"UNKNOWN"}.
        b2: Second belief vector {predicate: "YES"/"NO"/"UNKNOWN"}.

    Returns:
        Number of predicates where beliefs differ.
    """
    shared_keys = set(b1.keys()) & set(b2.keys())
    return sum(1 for k in shared_keys if b1[k] != b2[k])


def belief_divergence_curve(
    b_clean: List[Dict[str, str]],
    b_perturbed: List[Dict[str, str]],
) -> List[float]:
    """Compute belief divergence over turns between clean and perturbed runs.

    Args:
        b_clean: List of belief vectors from the clean (unperturbed) run.
        b_perturbed: List of belief vectors from the perturbed run.

    Returns:
        List of normalized divergence values (0.0 to 1.0) per turn.
        Each value is Hamming distance / number of shared predicates.
    """
    n_turns = min(len(b_clean), len(b_perturbed))
    curve: List[float] = []

    for t in range(n_turns):
        shared = set(b_clean[t].keys()) & set(b_perturbed[t].keys())
        if not shared:
            curve.append(0.0)
            continue
        dist = hamming_distance(b_clean[t], b_perturbed[t])
        curve.append(dist / len(shared))

    return curve


def instability_score(curves: List[List[float]]) -> float:
    """Compute instability score as mean area under divergence curves.

    The instability score measures how much a model's beliefs shift
    when inputs are perturbed. Higher scores indicate more instability.

    Args:
        curves: List of divergence curves (one per perturbation trial).

    Returns:
        Mean area under divergence curves, normalized by curve length.
        Returns 0.0 if no curves are provided.
    """
    if not curves:
        return 0.0

    areas: List[float] = []
    for curve in curves:
        if not curve:
            areas.append(0.0)
            continue
        area = sum(curve) / len(curve)
        areas.append(area)

    return sum(areas) / len(areas)


def sensitivity_profile(
    clean_beliefs: List[Dict[str, str]],
    perturbed_runs: Dict[str, List[Dict[str, str]]],
) -> Dict[str, float]:
    """Measure belief shift across multiple perturbation types.

    Args:
        clean_beliefs: Belief vectors from unperturbed evaluation.
        perturbed_runs: Dict mapping perturbation type name to
            belief vectors from that perturbation.

    Returns:
        Dict mapping perturbation type to instability score.
    """
    profile: Dict[str, float] = {}

    for perturb_type, perturbed_beliefs in perturbed_runs.items():
        curve = belief_divergence_curve(clean_beliefs, perturbed_beliefs)
        profile[perturb_type] = instability_score([curve])

    return profile


def belief_flip_rate(
    b_before: Dict[str, str],
    b_after: Dict[str, str],
) -> float:
    """Compute the fraction of shared predicates that flipped.

    Args:
        b_before: Belief vector before perturbation.
        b_after: Belief vector after perturbation.

    Returns:
        Fraction of shared predicates that changed value (0.0 to 1.0).
        Returns 0.0 if no shared predicates.
    """
    shared = set(b_before.keys()) & set(b_after.keys())
    if not shared:
        return 0.0
    flips = sum(1 for k in shared if b_before[k] != b_after[k])
    return flips / len(shared)


def correlation_instability_accuracy(
    instability_scores: List[float],
    accuracies: List[float],
) -> Optional[float]:
    """Compute Pearson correlation between instability and accuracy failure.

    Args:
        instability_scores: List of per-model instability scores.
        accuracies: List of per-model accuracy values.

    Returns:
        Pearson correlation coefficient, or None if insufficient data.
    """
    n = len(instability_scores)
    if n < 3 or n != len(accuracies):
        return None

    accuracy_failures = [1.0 - a for a in accuracies]

    mean_x = sum(instability_scores) / n
    mean_y = sum(accuracy_failures) / n

    cov = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(instability_scores, accuracy_failures)
    ) / n

    var_x = sum((x - mean_x) ** 2 for x in instability_scores) / n
    var_y = sum((y - mean_y) ** 2 for y in accuracy_failures) / n

    if var_x == 0 or var_y == 0:
        return None

    return cov / (var_x ** 0.5 * var_y ** 0.5)
