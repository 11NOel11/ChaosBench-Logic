"""Tests for instance-level selective guardrail policy."""

from __future__ import annotations

from chaosbench.repair.instance_policy import (
    apply_instance_policy,
    fit_instance_policy,
    score_instance_candidate,
)
from chaosbench.repair.selective import FlipCandidate


def test_instance_policy_keeps_beneficial_and_vetoes_harmful_cell() -> None:
    candidates = [
        FlipCandidate(
            0,
            "a0",
            "multi_hop",
            "s",
            "chaotic",
            1,
            "FALSE",
            "TRUE",
            0.10,
            4,
            1,
            0,
        ),
        FlipCandidate(
            1,
            "a1",
            "multi_hop",
            "s",
            "ergodic",
            1,
            "FALSE",
            "TRUE",
            0.20,
            4,
            1,
            0,
        ),
        FlipCandidate(
            2,
            "a2",
            "multi_hop",
            "s",
            "mixing",
            1,
            "TRUE",
            "FALSE",
            0.90,
            4,
            0,
            1,
        ),
    ]
    policy, _, _, _ = fit_instance_policy(
        candidates=candidates,
        margin_step=0.5,
        support_cap=8,
        shrinkage=0.0,
        min_family_samples=1,
        degrade_penalty=1.0,
    )

    repaired_records = [
        {
            "parsed_label": "FALSE",
            "repaired_label": "TRUE",
            "was_flipped": True,
            "flip_reason": "axiom_repair",
        },
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
    out, stats = apply_instance_policy(repaired_records, candidates, policy)

    assert out[0]["repaired_label"] == "TRUE"
    assert out[1]["repaired_label"] == "TRUE"
    assert out[2]["repaired_label"] == "TRUE"
    assert out[2]["flip_reason"] == "veto_instance_policy"
    assert stats["kept_flips"] == 2.0
    assert stats["vetoed_flips"] == 1.0


def test_instance_policy_disables_negative_family() -> None:
    candidates = [
        FlipCandidate(
            0,
            "b0",
            "fol_inference",
            "s",
            "deterministic",
            1,
            "TRUE",
            "FALSE",
            0.20,
            3,
            0,
            1,
        ),
        FlipCandidate(
            1,
            "b1",
            "fol_inference",
            "s",
            "aperiodic",
            1,
            "TRUE",
            "FALSE",
            0.30,
            3,
            0,
            1,
        ),
    ]
    policy, family_rows, _, _ = fit_instance_policy(
        candidates=candidates,
        margin_step=0.5,
        support_cap=8,
        shrinkage=0.0,
        min_family_samples=1,
        degrade_penalty=1.0,
    )

    by_family = {row["family"]: row for row in family_rows}
    assert by_family["fol_inference"]["enabled"] == 0.0

    repaired_records = [
        {
            "parsed_label": "TRUE",
            "repaired_label": "FALSE",
            "was_flipped": True,
            "flip_reason": "axiom_repair",
        }
    ]
    out, stats = apply_instance_policy(repaired_records, [candidates[0]], policy)
    assert out[0]["repaired_label"] == "TRUE"
    assert stats["kept_flips"] == 0.0
    assert stats["vetoed_flips"] == 1.0


def test_score_falls_back_to_family_prior_for_unseen_cell() -> None:
    train = [
        FlipCandidate(
            0,
            "c0",
            "consistency_paraphrase",
            "s",
            "chaotic",
            1,
            "FALSE",
            "TRUE",
            0.10,
            2,
            1,
            0,
        )
    ]
    policy, _, _, _ = fit_instance_policy(
        candidates=train,
        margin_step=0.5,
        support_cap=4,
        shrinkage=0.0,
        min_family_samples=1,
        degrade_penalty=1.0,
    )
    unseen = FlipCandidate(
        1,
        "c1",
        "consistency_paraphrase",
        "s",
        "mixing",
        1,
        "FALSE",
        "TRUE",
        0.95,
        20,
        0,
        0,
    )

    score = score_instance_candidate(unseen, policy)
    assert score == 1.0
