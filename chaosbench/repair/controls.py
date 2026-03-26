"""Control baselines and perturbation utilities for M1 analyses."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

VALID_BINARY_LABELS = {"TRUE", "FALSE"}


def flip_binary_label(label: str) -> str:
    """Flip TRUE/FALSE label."""
    if label == "TRUE":
        return "FALSE"
    if label == "FALSE":
        return "TRUE"
    raise ValueError(f"Unsupported binary label: {label}")


def valid_indices(
    records: Sequence[Dict], label_key: str = "parsed_label"
) -> List[int]:
    """Return indices with binary labels in a record sequence."""
    return [
        idx
        for idx, row in enumerate(records)
        if row.get(label_key) in VALID_BINARY_LABELS
    ]


def random_flip_labels(
    records: Sequence[Dict],
    target_flips: int,
    seed: int,
    label_key: str = "parsed_label",
) -> Tuple[List[str], int]:
    """Return labels after random budget-matched flips on valid rows."""
    labels = [str(row.get(label_key, "")) for row in records]
    valid = valid_indices(records, label_key=label_key)
    if not valid:
        return labels, 0

    budget = int(max(0, min(target_flips, len(valid))))
    if budget == 0:
        return labels, 0

    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(valid), size=budget, replace=False)
    for idx in sorted(chosen.tolist()):
        labels[idx] = flip_binary_label(labels[idx])
    return labels, budget


def budget_match_candidate_labels(
    records: Sequence[Dict],
    candidate_labels: Sequence[str],
    target_flips: int,
    seed: int,
    label_key: str = "parsed_label",
) -> Tuple[List[str], int, int, int]:
    """Match candidate signal to target flip budget deterministically.

    Returns a tuple:
      (labels, flips_applied, candidate_flips_used, synthetic_random_flips)
    """
    if len(records) != len(candidate_labels):
        raise ValueError("records and candidate_labels length mismatch")

    base = [str(row.get(label_key, "")) for row in records]
    out = list(base)
    valid = [idx for idx, value in enumerate(base) if value in VALID_BINARY_LABELS]
    budget = int(max(0, min(target_flips, len(valid))))
    if budget == 0:
        return out, 0, 0, 0

    candidate_changed = [
        idx
        for idx in valid
        if candidate_labels[idx] in VALID_BINARY_LABELS
        and candidate_labels[idx] != base[idx]
    ]

    rng = np.random.default_rng(seed)
    selected_from_candidate: List[int] = []
    synthetic_random: List[int] = []

    if len(candidate_changed) >= budget:
        chosen = rng.choice(np.asarray(candidate_changed), size=budget, replace=False)
        selected_from_candidate = sorted(chosen.tolist())
    else:
        selected_from_candidate = sorted(candidate_changed)
        remaining = budget - len(selected_from_candidate)
        pool = [idx for idx in valid if idx not in set(selected_from_candidate)]
        if remaining > 0 and pool:
            extra = rng.choice(
                np.asarray(pool), size=min(remaining, len(pool)), replace=False
            )
            synthetic_random = sorted(extra.tolist())

    for idx in selected_from_candidate:
        out[idx] = str(candidate_labels[idx])
    for idx in synthetic_random:
        out[idx] = flip_binary_label(out[idx])

    applied = len(selected_from_candidate) + len(synthetic_random)
    return out, applied, len(selected_from_candidate), len(synthetic_random)


def inject_parser_noise(
    records: Sequence[Dict],
    noise_rate: float,
    seed: int,
    label_key: str = "parsed_label",
) -> Tuple[List[Dict], int]:
    """Return shallow-copied records with random parser-label flips."""
    copied = [dict(row) for row in records]
    valid = [
        idx
        for idx, row in enumerate(copied)
        if row.get(label_key) in VALID_BINARY_LABELS
    ]
    if not valid:
        return copied, 0

    n_flip = int(round(max(0.0, min(1.0, noise_rate)) * len(valid)))
    n_flip = min(n_flip, len(valid))
    if n_flip == 0:
        return copied, 0

    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(valid), size=n_flip, replace=False)
    for idx in sorted(chosen.tolist()):
        copied[idx][label_key] = flip_binary_label(str(copied[idx][label_key]))

    return copied, n_flip


def shuffled_gate_families(
    observed_families: Sequence[str],
    gate_families: Sequence[str],
    seed: int,
) -> Tuple[str, ...]:
    """Create a wrong-family gate with same size as the original gate."""
    all_families = sorted({family for family in observed_families if family})
    if not all_families:
        return tuple()

    gate_set = set(gate_families)
    k = len(gate_set) if gate_set else min(3, len(all_families))
    k = min(k, len(all_families))

    pool = [family for family in all_families if family not in gate_set]
    if len(pool) < k:
        pool = list(all_families)

    rng = np.random.default_rng(seed)
    order = np.arange(len(pool))
    rng.shuffle(order)
    picked = sorted(pool[idx] for idx in order[:k].tolist())
    return tuple(picked)
