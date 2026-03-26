"""Tests for M5 cross-fit calibration helpers."""

from __future__ import annotations

from scripts.calibrate_m5_provider_thresholds import choose_threshold, threshold_grid


def test_choose_threshold_prefers_zero_negative_option() -> None:
    threshold_stats = [
        (
            0.05,
            {
                "mean_diff": 0.010,
                "mean_policy": 0.100,
                "mean_flip_policy": 0.02,
                "negatives": 1.0,
            },
        ),
        (
            0.20,
            {
                "mean_diff": 0.005,
                "mean_policy": 0.090,
                "mean_flip_policy": 0.01,
                "negatives": 0.0,
            },
        ),
        (
            0.30,
            {
                "mean_diff": 0.003,
                "mean_policy": 0.080,
                "mean_flip_policy": 0.01,
                "negatives": 0.0,
            },
        ),
    ]
    selected = choose_threshold(threshold_stats, default_threshold=0.05)
    assert selected == 0.20


def test_threshold_grid_includes_exact_max() -> None:
    values = threshold_grid(min_value=0.0, max_value=0.6, step=0.1)
    assert values[0] == 0.0
    assert values[-1] == 0.6
    assert len(values) == 7
