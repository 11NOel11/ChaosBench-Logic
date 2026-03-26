"""Tests for online safe-transfer controller."""

from __future__ import annotations

from chaosbench.repair.instance_policy import fit_instance_policy
from chaosbench.repair.online_controller import (
    OnlineControllerConfig,
    OnlineTransferController,
)
from chaosbench.repair.selective import FlipCandidate


def _train_candidates() -> list[FlipCandidate]:
    return [
        FlipCandidate(
            0,
            "id0",
            "multi_hop",
            "sys",
            "chaotic",
            1,
            "FALSE",
            "TRUE",
            0.1,
            4,
            1,
            0,
        ),
        FlipCandidate(
            1,
            "id1",
            "multi_hop",
            "sys",
            "mixing",
            1,
            "FALSE",
            "TRUE",
            0.2,
            4,
            1,
            0,
        ),
        FlipCandidate(
            2,
            "id2",
            "fol_inference",
            "sys",
            "deterministic",
            1,
            "TRUE",
            "FALSE",
            0.8,
            2,
            0,
            1,
        ),
    ]


def test_shift_score_detects_distribution_change() -> None:
    candidates = _train_candidates()
    policy, _, _, _ = fit_instance_policy(
        candidates,
        margin_step=0.5,
        support_cap=8,
        shrinkage=0.0,
        min_family_samples=1,
    )
    controller = OnlineTransferController.from_policy(
        policy,
        OnlineControllerConfig(),
    )

    same_shift = controller.shift_score(candidates, margin_step=0.5, support_cap=8)
    shifted = [
        FlipCandidate(
            10,
            "id10",
            "multi_hop",
            "sys",
            "chaotic",
            1,
            "FALSE",
            "TRUE",
            1.0,
            8,
            0,
            0,
        )
        for _ in range(6)
    ]
    high_shift = controller.shift_score(shifted, margin_step=0.5, support_cap=8)

    assert same_shift <= high_shift
    assert high_shift > 0.0


def test_online_update_tightens_after_harm() -> None:
    policy, _, _, _ = fit_instance_policy(
        _train_candidates(),
        margin_step=0.5,
        support_cap=8,
        shrinkage=0.0,
        min_family_samples=1,
    )
    controller = OnlineTransferController.from_policy(
        policy,
        OnlineControllerConfig(
            threshold_step=0.2,
            risk_budget_b0=0.0,
            alarm_threshold=1.0,
        ),
    )

    update = controller.update(
        provider="openai/gpt-4o",
        fallback_threshold=0.2,
        shift_score=0.3,
        delta_mcc=-0.02,
        baseline_axiom_rate=0.1,
        policy_axiom_rate=0.2,
    )
    assert update["threshold_after"] >= update["threshold_before"]
    assert update["harm_loss"] > 0.0


def test_online_update_relaxes_after_benefit() -> None:
    policy, _, _, _ = fit_instance_policy(
        _train_candidates(),
        margin_step=0.5,
        support_cap=8,
        shrinkage=0.0,
        min_family_samples=1,
    )
    controller = OnlineTransferController.from_policy(
        policy,
        OnlineControllerConfig(threshold_step=0.2),
    )
    controller.threshold_for_provider("openai/gpt-4o", fallback_threshold=0.6)

    update = controller.update(
        provider="openai/gpt-4o",
        fallback_threshold=0.6,
        shift_score=0.0,
        delta_mcc=0.03,
        baseline_axiom_rate=0.2,
        policy_axiom_rate=0.2,
    )
    assert update["threshold_after"] <= update["threshold_before"]
    assert update["utility"] > 0.0


def test_online_update_triggers_alarm() -> None:
    policy, _, _, _ = fit_instance_policy(
        _train_candidates(),
        margin_step=0.5,
        support_cap=8,
        shrinkage=0.0,
        min_family_samples=1,
    )
    controller = OnlineTransferController.from_policy(
        policy,
        OnlineControllerConfig(
            threshold_step=0.05,
            risk_budget_b0=0.0,
            alarm_threshold=0.001,
            emergency_step=0.2,
        ),
    )

    update = controller.update(
        provider="openrouter/meta-llama",
        fallback_threshold=0.1,
        shift_score=0.5,
        delta_mcc=-0.05,
        baseline_axiom_rate=0.1,
        policy_axiom_rate=0.2,
    )
    assert update["alarm_triggered"] == 1.0
    assert update["threshold_after"] > update["threshold_before"]


def test_online_sweep_tightens_when_local_risk_is_high() -> None:
    policy, _, _, _ = fit_instance_policy(
        _train_candidates(),
        margin_step=0.5,
        support_cap=8,
        shrinkage=0.0,
        min_family_samples=1,
    )
    controller = OnlineTransferController.from_policy(
        policy,
        OnlineControllerConfig(
            threshold_step=0.1,
            sweep_radius=2,
            sweep_mix=1.0,
            eta0=0.0,
        ),
    )

    update = controller.update(
        provider="openrouter/meta-llama",
        fallback_threshold=0.2,
        shift_score=0.0,
        delta_mcc=0.0,
        baseline_axiom_rate=0.0,
        policy_axiom_rate=0.0,
        candidate_rows=[
            {"score": 0.2, "improved": 1.0, "degraded": 0.0, "enabled": 1.0},
            {"score": 0.2, "improved": 0.0, "degraded": 1.0, "enabled": 1.0},
            {"score": 0.3, "improved": 0.0, "degraded": 1.0, "enabled": 1.0},
            {"score": 0.4, "improved": 0.0, "degraded": 1.0, "enabled": 1.0},
        ],
        degrade_penalty=2.0,
    )

    assert update["sweep_applied"] == 1.0
    assert update["sweep_threshold"] > update["threshold_after_primal"]
    assert update["threshold_after"] == update["sweep_threshold"]


def test_provider_specific_reference_overrides_global_shift() -> None:
    train = _train_candidates()
    policy, _, _, _ = fit_instance_policy(
        train,
        margin_step=0.5,
        support_cap=8,
        shrinkage=0.0,
        min_family_samples=1,
    )
    shifted = [
        FlipCandidate(
            30,
            "id30",
            "multi_hop",
            "sys",
            "chaotic",
            1,
            "FALSE",
            "TRUE",
            1.0,
            8,
            0,
            0,
        )
        for _ in range(5)
    ]
    provider_ref = {
        "openrouter": {
            "multi_hop|m2|s8": 1.0,
        }
    }
    controller = OnlineTransferController.from_policy(
        policy,
        OnlineControllerConfig(),
        provider_reference_dists=provider_ref,
    )

    global_shift = controller.shift_score(
        shifted,
        margin_step=0.5,
        support_cap=8,
        provider="unknown/provider",
    )
    provider_shift = controller.shift_score(
        shifted,
        margin_step=0.5,
        support_cap=8,
        provider="openrouter/meta-llama",
    )

    assert provider_shift < global_shift
    assert provider_shift == 0.0


def test_provider_step_multiplier_dampens_update_size() -> None:
    policy, _, _, _ = fit_instance_policy(
        _train_candidates(),
        margin_step=0.5,
        support_cap=8,
        shrinkage=0.0,
        min_family_samples=1,
    )
    base = OnlineTransferController.from_policy(
        policy,
        OnlineControllerConfig(threshold_step=0.2),
    )
    damped = OnlineTransferController.from_policy(
        policy,
        OnlineControllerConfig(
            threshold_step=0.2,
            provider_step_multipliers={"openai": 0.25},
        ),
    )

    base.threshold_for_provider("openai/gpt-4o", fallback_threshold=0.6)
    damped.threshold_for_provider("openai/gpt-4o", fallback_threshold=0.6)

    base_update = base.update(
        provider="openai/gpt-4o",
        fallback_threshold=0.6,
        shift_score=0.0,
        delta_mcc=0.03,
        baseline_axiom_rate=0.0,
        policy_axiom_rate=0.0,
    )
    damped_update = damped.update(
        provider="openai/gpt-4o",
        fallback_threshold=0.6,
        shift_score=0.0,
        delta_mcc=0.03,
        baseline_axiom_rate=0.0,
        policy_axiom_rate=0.0,
    )

    assert damped_update["step_multiplier"] == 0.25
    assert abs(damped_update["delta_threshold"]) < abs(base_update["delta_threshold"])


def test_non_degrade_guard_blocks_relaxation_after_regression() -> None:
    policy, _, _, _ = fit_instance_policy(
        _train_candidates(),
        margin_step=0.5,
        support_cap=8,
        shrinkage=0.0,
        min_family_samples=1,
    )
    controller = OnlineTransferController.from_policy(
        policy,
        OnlineControllerConfig(
            threshold_step=0.1,
            sweep_radius=2,
            sweep_mix=1.0,
            eta0=0.0,
            non_degrade_rollback_step=0.05,
        ),
    )

    update = controller.update(
        provider="openai/gpt-4o",
        fallback_threshold=0.4,
        shift_score=0.0,
        delta_mcc=-0.01,
        baseline_axiom_rate=0.0,
        policy_axiom_rate=0.0,
        candidate_rows=[
            {"score": 0.2, "improved": 1.0, "degraded": 0.0, "enabled": 1.0},
            {"score": 0.3, "improved": 1.0, "degraded": 0.0, "enabled": 1.0},
            {"score": 0.4, "improved": 1.0, "degraded": 0.0, "enabled": 1.0},
        ],
    )

    assert update["sweep_threshold"] < update["threshold_before"]
    assert update["non_degrade_guard_triggered"] == 1.0
    assert update["threshold_after"] >= update["threshold_before"]
    assert update["non_degrade_rollback_applied"] > 0.0


def test_sweep_tie_break_prefers_higher_threshold() -> None:
    policy, _, _, _ = fit_instance_policy(
        _train_candidates(),
        margin_step=0.5,
        support_cap=8,
        shrinkage=0.0,
        min_family_samples=1,
    )
    controller = OnlineTransferController.from_policy(
        policy,
        OnlineControllerConfig(
            threshold_step=0.1,
            sweep_radius=1,
            sweep_mix=1.0,
            eta0=0.0,
        ),
    )
    controller.threshold_for_provider("openai/gpt-4o", fallback_threshold=0.3)

    update = controller.update(
        provider="openai/gpt-4o",
        fallback_threshold=0.3,
        shift_score=0.0,
        delta_mcc=0.0,
        baseline_axiom_rate=0.0,
        policy_axiom_rate=0.0,
        candidate_rows=[
            {"score": 0.0, "improved": 0.0, "degraded": 0.0, "enabled": 0.0}
        ],
    )

    assert update["sweep_objective_gain"] == 0.0
    assert update["sweep_threshold"] == 0.4
    assert update["threshold_after"] == 0.4


def test_sweep_requires_min_objective_gain_to_move() -> None:
    policy, _, _, _ = fit_instance_policy(
        _train_candidates(),
        margin_step=0.5,
        support_cap=8,
        shrinkage=0.0,
        min_family_samples=1,
    )
    controller = OnlineTransferController.from_policy(
        policy,
        OnlineControllerConfig(
            threshold_step=0.1,
            sweep_radius=1,
            sweep_mix=1.0,
            sweep_min_improvement=0.01,
            eta0=0.0,
        ),
    )
    controller.threshold_for_provider("openai/gpt-4o", fallback_threshold=0.3)

    update = controller.update(
        provider="openai/gpt-4o",
        fallback_threshold=0.3,
        shift_score=0.0,
        delta_mcc=0.0,
        baseline_axiom_rate=0.0,
        policy_axiom_rate=0.0,
        candidate_rows=[
            {"score": 0.0, "improved": 0.0, "degraded": 0.0, "enabled": 0.0}
        ],
    )

    assert update["sweep_objective_gain"] == 0.0
    assert update["sweep_applied"] == 0.0
    assert update["threshold_after"] == 0.3
