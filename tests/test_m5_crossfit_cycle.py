from __future__ import annotations

import csv

import pytest

from scripts.run_m5_crossfit_cycle import (
    build_temporal_splits,
    apply_provider_threshold_offsets,
    parse_provider_threshold_offsets,
    slugify,
    summarize_paired_transfer,
)


def test_parse_provider_threshold_offsets_basic() -> None:
    parsed = parse_provider_threshold_offsets("openai=-0.05, gemini=0.02")
    assert parsed == {"openai": -0.05, "gemini": 0.02}


def test_parse_provider_threshold_offsets_empty() -> None:
    assert parse_provider_threshold_offsets("") == {}


def test_apply_provider_threshold_offsets_clips() -> None:
    base = {"openai": 0.11, "gemini": 0.54, "deepseek": 0.02}
    shifted = apply_provider_threshold_offsets(
        provider_map=base,
        offsets={"openai": -0.5, "gemini": 2.0},
        threshold_min=0.0,
        threshold_max=1.5,
    )
    assert shifted["openai"] == 0.0
    assert shifted["gemini"] == 1.5
    assert shifted["deepseek"] == 0.02


def test_slugify_normalizes_and_collapses() -> None:
    assert slugify("OpenRouter Live/v1") == "openrouter_live_v1"


def test_build_temporal_splits_respects_constraints() -> None:
    cuts = build_temporal_splits(
        run_ids=["r0", "r1", "r2", "r3"],
        min_train_runs=2,
        min_test_runs=1,
        max_cuts=0,
    )
    assert [cut for cut, _ in cuts] == [1, 2]
    first_rows = cuts[0][1]
    assert first_rows[0]["split"] == "dev"
    assert first_rows[2]["split"] == "heldout"


def test_summarize_paired_transfer_with_split_filter(tmp_path) -> None:
    static_csv = tmp_path / "static.csv"
    online_csv = tmp_path / "online.csv"

    with static_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "split",
                "delta_mcc_baseline",
                "delta_mcc_policy",
                "policy_minus_baseline_mcc",
                "harm_loss",
                "alarm_triggered",
                "shift_score",
                "update_applied",
                "row_flip_rate_policy",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "a",
                "split": "heldout",
                "delta_mcc_baseline": 0.0,
                "delta_mcc_policy": 0.1,
                "policy_minus_baseline_mcc": 0.1,
                "harm_loss": 0.0,
                "alarm_triggered": 0.0,
                "shift_score": 0.0,
                "update_applied": 0.0,
                "row_flip_rate_policy": 0.0,
            }
        )
        writer.writerow(
            {
                "run_id": "b",
                "split": "dev",
                "delta_mcc_baseline": 0.0,
                "delta_mcc_policy": 0.2,
                "policy_minus_baseline_mcc": 0.2,
                "harm_loss": 0.0,
                "alarm_triggered": 0.0,
                "shift_score": 0.0,
                "update_applied": 0.0,
                "row_flip_rate_policy": 0.0,
            }
        )

    with online_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "split",
                "delta_mcc_baseline",
                "delta_mcc_policy",
                "policy_minus_baseline_mcc",
                "harm_loss",
                "alarm_triggered",
                "shift_score",
                "update_applied",
                "row_flip_rate_policy",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "a",
                "split": "heldout",
                "delta_mcc_baseline": 0.0,
                "delta_mcc_policy": 0.12,
                "policy_minus_baseline_mcc": 0.12,
                "harm_loss": 0.0,
                "alarm_triggered": 0.0,
                "shift_score": 0.0,
                "update_applied": 0.0,
                "row_flip_rate_policy": 0.0,
            }
        )
        writer.writerow(
            {
                "run_id": "b",
                "split": "dev",
                "delta_mcc_baseline": 0.0,
                "delta_mcc_policy": 0.5,
                "policy_minus_baseline_mcc": 0.5,
                "harm_loss": 0.0,
                "alarm_triggered": 0.0,
                "shift_score": 0.0,
                "update_applied": 1.0,
                "row_flip_rate_policy": 0.0,
            }
        )

    stats = summarize_paired_transfer(
        static_csv=static_csv,
        online_csv=online_csv,
        split_filter="heldout",
    )
    assert stats is not None
    assert stats["n"] == 1.0
    assert stats["mean_online_minus_static"] == pytest.approx(0.02)
