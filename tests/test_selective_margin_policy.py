"""Tests for selective margin-aware guardrail policy."""

from __future__ import annotations

from chaosbench.repair import RepairConfig, repair_records
from chaosbench.repair.selective import (
    FlipCandidate,
    apply_margin_policy,
    collect_flip_candidates,
    fit_margin_policy,
    policy_map,
)


def test_fit_margin_policy_enables_only_positive_family():
    candidates = [
        FlipCandidate(
            0, "a0", "multi_hop", "s", "chaotic", 1, "FALSE", "TRUE", 0.1, 5, 1, 0
        ),
        FlipCandidate(
            1, "a1", "multi_hop", "s", "mixing", 1, "FALSE", "TRUE", 0.4, 4, 1, 0
        ),
        FlipCandidate(
            2, "a2", "multi_hop", "s", "ergodic", 1, "TRUE", "FALSE", 0.9, 3, 0, 1
        ),
        FlipCandidate(
            3,
            "b0",
            "fol_inference",
            "s",
            "deterministic",
            1,
            "TRUE",
            "FALSE",
            0.2,
            4,
            0,
            1,
        ),
    ]

    rows = fit_margin_policy(
        candidates=candidates,
        threshold_step=0.5,
        min_family_samples=1,
        min_support=1,
    )
    by_family = policy_map(rows)

    assert by_family["multi_hop"]["enabled"] == 1.0
    assert by_family["multi_hop"]["threshold"] == 0.5
    assert by_family["fol_inference"]["enabled"] == 0.0


def test_apply_margin_policy_vetoes_when_family_disabled_or_margin_high():
    repaired_records = [
        {
            "parsed_label": "FALSE",
            "repaired_label": "TRUE",
            "was_flipped": True,
            "flip_reason": "axiom_repair",
        },
        {
            "parsed_label": "TRUE",
            "repaired_label": "FALSE",
            "was_flipped": True,
            "flip_reason": "axiom_repair",
        },
    ]
    candidates = [
        FlipCandidate(
            0, "a0", "multi_hop", "s", "chaotic", 1, "FALSE", "TRUE", 0.2, 3, 1, 0
        ),
        FlipCandidate(
            1,
            "b0",
            "fol_inference",
            "s",
            "deterministic",
            1,
            "TRUE",
            "FALSE",
            0.2,
            3,
            0,
            1,
        ),
    ]
    policy = {
        "multi_hop": {"enabled": 1.0, "threshold": 0.3, "min_support": 2.0},
        "fol_inference": {"enabled": 0.0, "threshold": -1.0, "min_support": 2.0},
    }

    out, stats = apply_margin_policy(
        repaired_records=repaired_records,
        candidates=candidates,
        per_family_policy=policy,
    )

    assert out[0]["repaired_label"] == "TRUE"
    assert out[1]["repaired_label"] == "TRUE"
    assert out[1]["flip_reason"] == "veto_margin_policy"
    assert stats["vetoed_flips"] == 1.0


def test_collect_candidates_matches_baseline_flip_indices():
    records = [
        {
            "id": "syn_chaotic",
            "question": "Is the synthetic system chaotic?",
            "ground_truth": "TRUE",
            "parsed_label": "TRUE",
            "task_family": "multi_hop",
            "outcome": "VALID_TRUE",
        },
        {
            "id": "syn_deterministic",
            "question": "Is the synthetic system deterministic?",
            "ground_truth": "TRUE",
            "parsed_label": "FALSE",
            "task_family": "multi_hop",
            "outcome": "VALID_FALSE",
        },
        {
            "id": "syn_random",
            "question": "Is the synthetic system random?",
            "ground_truth": "FALSE",
            "parsed_label": "TRUE",
            "task_family": "multi_hop",
            "outcome": "VALID_TRUE",
        },
    ]
    id_to_meta = {
        "syn_chaotic": {"system_id": "synthetic_system", "task_family": "multi_hop"},
        "syn_deterministic": {
            "system_id": "synthetic_system",
            "task_family": "multi_hop",
        },
        "syn_random": {"system_id": "synthetic_system", "task_family": "multi_hop"},
    }
    config = RepairConfig(
        name="synthetic_selective",
        gate_families=("multi_hop",),
        extractor_strategy="last_mention",
        polarity_mode="none",
    )

    baseline = repair_records(records=records, id_to_meta=id_to_meta, config=config)
    candidates = collect_flip_candidates(
        records=records, id_to_meta=id_to_meta, config=config
    )

    flipped_indices = {
        index
        for index, row in enumerate(baseline.records)
        if row.get("was_flipped") is True
    }
    candidate_indices = {cand.row_index for cand in candidates}
    assert candidate_indices == flipped_indices
