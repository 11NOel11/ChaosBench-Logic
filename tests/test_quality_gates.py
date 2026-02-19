"""Tests for the quality gates module."""

import pytest

from chaosbench.quality.gates import (
    check_near_duplicates,
    check_label_leakage,
    check_class_balance,
    check_difficulty_distribution,
    run_all_gates,
    GateResult,
)


def _make_item(item_id="q1", question="Is Lorenz chaotic?", gt="TRUE", qtype="atomic", system_id="lorenz63"):
    return {
        "id": item_id,
        "question": question,
        "ground_truth": gt,
        "type": qtype,
        "system_id": system_id,
    }


class TestNearDuplicates:
    def test_no_duplicates(self):
        items = [
            _make_item("q1", "Is Lorenz chaotic?"),
            _make_item("q2", "Is Rossler periodic?"),
            _make_item("q3", "Is SHM deterministic?"),
        ]
        result = check_near_duplicates(items)
        assert result.passed
        assert result.stats["exact_duplicates"] == 0

    def test_exact_duplicates_detected(self):
        items = [
            _make_item("q1", "Is Lorenz chaotic?"),
            _make_item("q2", "Is Lorenz chaotic?"),
        ]
        result = check_near_duplicates(items, max_allowed=0)
        assert not result.passed
        assert result.stats["exact_duplicates"] == 1

    def test_consistency_paraphrase_excluded(self):
        items = [
            _make_item("q1", "Is Lorenz chaotic?", qtype="consistency_paraphrase"),
            _make_item("q2", "Is Lorenz chaotic?", qtype="consistency_paraphrase"),
        ]
        result = check_near_duplicates(items, max_allowed=0)
        assert result.passed  # Excluded from checking

    def test_near_duplicates_by_jaccard(self):
        items = [
            _make_item("q1", "Is the Lorenz system chaotic?"),
            _make_item("q2", "Is the Lorenz system really chaotic?"),
        ]
        result = check_near_duplicates(items, jaccard_threshold=0.7, max_allowed=0)
        # These are near-duplicates with high Jaccard similarity
        assert result.stats["items_checked"] == 2


class TestLabelLeakage:
    def test_no_leakage(self):
        items = [
            _make_item("q1", "Is Lorenz chaotic?"),
        ]
        result = check_label_leakage(items)
        assert result.passed

    def test_forbidden_token_detected(self):
        items = [
            _make_item("q1", "The ground_truth is that Lorenz is chaotic"),
        ]
        result = check_label_leakage(items)
        assert not result.passed
        assert len(result.violations) == 1

    def test_answer_echo_detected(self):
        items = [
            _make_item("q1", "This is true: Lorenz is chaotic", gt="TRUE"),
        ]
        result = check_label_leakage(items)
        assert not result.passed


class TestClassBalance:
    def test_balanced(self):
        items = [
            _make_item("q1", gt="TRUE"),
            _make_item("q2", gt="FALSE"),
        ]
        result = check_class_balance(items, min_items=1)
        assert result.passed
        assert abs(result.stats["overall_true_ratio"] - 0.5) < 0.01

    def test_imbalanced(self):
        items = [_make_item(f"q{i}", gt="TRUE") for i in range(20)]
        result = check_class_balance(items, balance_range=(0.35, 0.65), min_items=10)
        assert not result.passed

    def test_small_family_skipped(self):
        items = [_make_item(f"q{i}", gt="TRUE") for i in range(5)]
        result = check_class_balance(items, min_items=20)
        assert result.passed  # Too few items, skip balance check


class TestDifficultyDistribution:
    def test_always_passes(self):
        items = [
            _make_item("q1", "Is Lorenz chaotic?", qtype="atomic"),
            _make_item("q2", "Multi-hop chain question", qtype="multi_hop"),
        ]
        result = check_difficulty_distribution(items)
        assert result.passed  # Always passes (report only)

    def test_reports_per_split(self):
        items = [
            {**_make_item("q1", qtype="atomic"), "_split": "core"},
            {**_make_item("q2", qtype="multi_hop"), "_split": "hard"},
        ]
        result = check_difficulty_distribution(items)
        assert "per_split_avg" in result.stats


class TestRunAllGates:
    def test_runs_all_four_gates(self):
        items = [
            _make_item("q1", "Is Lorenz chaotic?", gt="TRUE"),
            _make_item("q2", "Is Rossler periodic?", gt="FALSE"),
        ]
        results = run_all_gates(items)
        assert len(results) == 4
        names = {r.gate_name for r in results}
        assert "near_duplicate_detection" in names
        assert "label_leakage_scan" in names
        assert "class_balance" in names
        assert "difficulty_distribution" in names

    def test_config_override(self):
        items = [
            _make_item("q1", "Is Lorenz chaotic?", gt="TRUE"),
            _make_item("q2", "Is Lorenz chaotic?", gt="TRUE"),
        ]
        # With strict config allowing 0 duplicates
        results = run_all_gates(items, {"max_near_duplicates": 0})
        dup_result = [r for r in results if r.gate_name == "near_duplicate_detection"][0]
        assert not dup_result.passed
