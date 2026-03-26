"""Tests for M1 control baselines and noise utilities."""

from __future__ import annotations

from chaosbench.repair.controls import (
    budget_match_candidate_labels,
    inject_parser_noise,
    random_flip_labels,
    shuffled_gate_families,
)


def _toy_records():
    return [
        {"parsed_label": "TRUE", "task_family": "multi_hop"},
        {"parsed_label": "FALSE", "task_family": "fol_inference"},
        {"parsed_label": "TRUE", "task_family": "atomic"},
        {"parsed_label": "FALSE", "task_family": "consistency_paraphrase"},
        {"parsed_label": "TRUE", "task_family": "perturbation"},
    ]


def _count_flips(records, labels):
    flips = 0
    for row, label in zip(records, labels):
        if row["parsed_label"] != label:
            flips += 1
    return flips


def test_random_flip_labels_matches_budget_and_is_deterministic():
    records = _toy_records()
    labels_a, flips_a = random_flip_labels(records, target_flips=2, seed=11)
    labels_b, flips_b = random_flip_labels(records, target_flips=2, seed=11)

    assert flips_a == 2
    assert flips_b == 2
    assert labels_a == labels_b
    assert _count_flips(records, labels_a) == 2


def test_budget_match_candidate_labels_enforces_exact_budget():
    records = _toy_records()
    candidate = ["FALSE", "TRUE", "TRUE", "FALSE", "TRUE"]
    labels, applied, used, synthetic = budget_match_candidate_labels(
        records,
        candidate_labels=candidate,
        target_flips=3,
        seed=23,
    )

    assert applied == 3
    assert _count_flips(records, labels) == 3
    assert used + synthetic == applied


def test_inject_parser_noise_flips_expected_count():
    records = _toy_records()
    noisy, n_flip = inject_parser_noise(records, noise_rate=0.4, seed=5)

    assert n_flip == 2
    assert _count_flips(records, [row["parsed_label"] for row in noisy]) == 2


def test_shuffled_gate_families_preserves_size_and_changes_membership():
    observed = [row["task_family"] for row in _toy_records()]
    gate = ("multi_hop", "fol_inference")
    shuffled = shuffled_gate_families(observed, gate_families=gate, seed=7)

    assert len(shuffled) == len(gate)
    assert set(shuffled) != set(gate)
