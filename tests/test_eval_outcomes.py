"""Tests for 3-way evaluation outcomes (VALID_CORRECT / VALID_INCORRECT / INVALID)."""

import pytest
from chaosbench.eval.metrics import (
    EvalResult,
    Outcome,
    normalize_label,
    _compute_flip_rate,
    compute_balanced_accuracy,
    compute_mcc,
    compute_per_class_metrics,
)


def test_normalize_label_valid():
    """Valid outputs should parse to YES/NO."""
    assert normalize_label("YES") == "YES"
    assert normalize_label("NO") == "NO"
    assert normalize_label("True") == "YES"
    assert normalize_label("False") == "NO"
    assert normalize_label("yes.") == "YES"
    assert normalize_label("no.") == "NO"
    assert normalize_label("The answer is YES") == "YES"
    assert normalize_label("The answer is NO") == "NO"


def test_normalize_label_invalid():
    """Invalid outputs should return None."""
    assert normalize_label("Maybe") is None
    assert normalize_label("I don't know") is None
    assert normalize_label("") is None
    assert normalize_label(None) is None
    assert normalize_label("12345") is None


def test_outcome_assignment():
    """Outcomes should be assigned correctly based on parsing."""
    # VALID_CORRECT
    r1 = EvalResult(
        item_id="q001",
        batch_file="batch8.jsonl",
        task_family="atomic",
        bias_family=None,
        dialogue_id=None,
        turn_index=None,
        system_id="lorenz63",
        gold="YES",
        pred_raw="YES",
        pred_norm="YES",
        correct=True,
        outcome=Outcome.VALID_CORRECT,
    )
    assert r1.outcome == Outcome.VALID_CORRECT

    # VALID_INCORRECT
    r2 = EvalResult(
        item_id="q002",
        batch_file="batch8.jsonl",
        task_family="atomic",
        bias_family=None,
        dialogue_id=None,
        turn_index=None,
        system_id="lorenz63",
        gold="YES",
        pred_raw="NO",
        pred_norm="NO",
        correct=False,
        outcome=Outcome.VALID_INCORRECT,
    )
    assert r2.outcome == Outcome.VALID_INCORRECT

    # INVALID
    r3 = EvalResult(
        item_id="q003",
        batch_file="batch8.jsonl",
        task_family="atomic",
        bias_family=None,
        dialogue_id=None,
        turn_index=None,
        system_id="lorenz63",
        gold="YES",
        pred_raw="I don't know",
        pred_norm=None,
        correct=None,
        outcome=Outcome.INVALID,
    )
    assert r3.outcome == Outcome.INVALID


def test_flip_rate_no_groups():
    """Flip rate should be None if no groups present."""
    results = [
        EvalResult(
            item_id="q001",
            batch_file="batch8.jsonl",
            task_family="atomic",
            bias_family=None,
            dialogue_id=None,
            turn_index=None,
            system_id="lorenz63",
            gold="YES",
            pred_raw="YES",
            pred_norm="YES",
            correct=True,
            group_id=None,
        ),
    ]
    assert _compute_flip_rate(results) is None


def test_flip_rate_with_groups():
    """Flip rate should count groups with disagreements."""
    results = [
        EvalResult(
            item_id="q001",
            batch_file="batch17.jsonl",
            task_family="perturbation",
            bias_family=None,
            dialogue_id=None,
            turn_index=None,
            system_id="lorenz63",
            gold="YES",
            pred_raw="YES",
            pred_norm="YES",
            correct=True,
            group_id="perturb_chaotic_core_0001",
        ),
        EvalResult(
            item_id="q002",
            batch_file="batch17.jsonl",
            task_family="perturbation",
            bias_family=None,
            dialogue_id=None,
            turn_index=None,
            system_id="lorenz63",
            gold="YES",
            pred_raw="NO",
            pred_norm="NO",
            correct=False,
            group_id="perturb_chaotic_core_0001",  # Same group, different answer
        ),
        EvalResult(
            item_id="q003",
            batch_file="batch17.jsonl",
            task_family="perturbation",
            bias_family=None,
            dialogue_id=None,
            turn_index=None,
            system_id="vdp",
            gold="NO",
            pred_raw="NO",
            pred_norm="NO",
            correct=True,
            group_id="perturb_chaotic_core_0002",  # Different group, no flip
        ),
        EvalResult(
            item_id="q004",
            batch_file="batch17.jsonl",
            task_family="perturbation",
            bias_family=None,
            dialogue_id=None,
            turn_index=None,
            system_id="vdp",
            gold="NO",
            pred_raw="NO",
            pred_norm="NO",
            correct=True,
            group_id="perturb_chaotic_core_0002",
        ),
    ]

    flip_rate = _compute_flip_rate(results)
    assert flip_rate == 0.5  # 1 flipped group out of 2 groups


def test_balanced_accuracy():
    """Balanced accuracy should average per-class recall."""
    results = [
        # 2/3 YES correct
        EvalResult("q1", "", "atomic", None, None, None, "l63", "YES", "", "YES", True),
        EvalResult("q2", "", "atomic", None, None, None, "l63", "YES", "", "YES", True),
        EvalResult("q3", "", "atomic", None, None, None, "l63", "YES", "", "NO", False),
        # 1/2 NO correct
        EvalResult("q4", "", "atomic", None, None, None, "l63", "NO", "", "NO", True),
        EvalResult("q5", "", "atomic", None, None, None, "l63", "NO", "", "YES", False),
    ]

    bal_acc = compute_balanced_accuracy(results)
    assert bal_acc is not None
    expected = (2/3 + 1/2) / 2  # (TPR + TNR) / 2 = (0.667 + 0.5) / 2 = 0.583
    assert abs(bal_acc - expected) < 0.01


def test_mcc():
    """MCC should compute Matthews correlation coefficient."""
    results = [
        # TP=2, TN=1, FP=1, FN=1
        EvalResult("q1", "", "atomic", None, None, None, "l63", "YES", "", "YES", True),  # TP
        EvalResult("q2", "", "atomic", None, None, None, "l63", "YES", "", "YES", True),  # TP
        EvalResult("q3", "", "atomic", None, None, None, "l63", "YES", "", "NO", False),  # FN
        EvalResult("q4", "", "atomic", None, None, None, "l63", "NO", "", "NO", True),  # TN
        EvalResult("q5", "", "atomic", None, None, None, "l63", "NO", "", "YES", False),  # FP
    ]

    mcc = compute_mcc(results)
    assert mcc is not None
    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    # MCC = (2*1 - 1*1) / sqrt(3*3*2*2) = 1 / sqrt(36) = 1/6 ≈ 0.167
    expected = (2*1 - 1*1) / ((3*3*2*2) ** 0.5)
    assert abs(mcc - expected) < 0.01


def test_per_class_metrics():
    """Per-class metrics should compute precision/recall/F1."""
    results = [
        EvalResult("q1", "", "atomic", None, None, None, "l63", "YES", "", "YES", True),  # TP
        EvalResult("q2", "", "atomic", None, None, None, "l63", "YES", "", "YES", True),  # TP
        EvalResult("q3", "", "atomic", None, None, None, "l63", "YES", "", "NO", False),  # FN
        EvalResult("q4", "", "atomic", None, None, None, "l63", "NO", "", "NO", True),  # TN
        EvalResult("q5", "", "atomic", None, None, None, "l63", "NO", "", "YES", False),  # FP
    ]

    metrics = compute_per_class_metrics(results)
    assert "YES" in metrics
    assert "NO" in metrics

    # YES: TP=2, FP=1, FN=1
    # Precision = TP/(TP+FP) = 2/3 = 0.667
    # Recall = TP/(TP+FN) = 2/3 = 0.667
    # F1 = 2*P*R/(P+R) = 2*0.667*0.667/(0.667+0.667) = 0.667
    assert abs(metrics["YES"]["precision"] - 2/3) < 0.01
    assert abs(metrics["YES"]["recall"] - 2/3) < 0.01
    assert abs(metrics["YES"]["f1"] - 2/3) < 0.01
    assert metrics["YES"]["support"] == 3

    # NO: TP=1, FP=1, FN=1
    # Precision = 1/2 = 0.5
    # Recall = 1/2 = 0.5
    # F1 = 0.5
    assert abs(metrics["NO"]["precision"] - 0.5) < 0.01
    assert abs(metrics["NO"]["recall"] - 0.5) < 0.01
    assert abs(metrics["NO"]["f1"] - 0.5) < 0.01
    assert metrics["NO"]["support"] == 2
