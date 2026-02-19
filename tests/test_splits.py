"""Tests for the 5-way split protocol."""

import json
import os
import tempfile

import pytest

from chaosbench.data.splits import (
    SPLIT_ASSIGNMENTS,
    VALID_SPLITS,
    assign_splits,
    compute_split_stats,
    get_split_for_batch,
    get_split_items,
    validate_splits,
)


def test_split_assignments_cover_known_batches():
    """All known batches have a split assignment."""
    known_batches = [
        "batch1_atomic_implication",
        "batch2_multiHop_crossSystem",
        "batch8_indicator_diagnostics",
        "batch10_adversarial",
        "batch11_consistency_paraphrase",
        "batch15_atomic_dysts",
        "batch16_multi_hop_dysts",
        "batch18_fol_dysts",
    ]
    for batch in known_batches:
        assert batch in SPLIT_ASSIGNMENTS, f"{batch} missing from SPLIT_ASSIGNMENTS"


def test_get_split_for_batch():
    """get_split_for_batch returns correct split names."""
    assert get_split_for_batch("batch1_atomic_implication") == "core"
    assert get_split_for_batch("batch10_adversarial") == "hard"
    assert get_split_for_batch("batch15_atomic_dysts") == "heldout_systems"
    assert get_split_for_batch("batch11_consistency_paraphrase") == "robustness"


def test_get_split_for_unknown_batch():
    """get_split_for_batch raises ValueError for unknown batch."""
    with pytest.raises(ValueError, match="Unknown batch"):
        get_split_for_batch("batch999_nonexistent")


def _make_test_data_dir(tmp_path):
    """Create a minimal data directory with test items."""
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)

    # Core batch
    items_core = [
        {"id": "core_1", "question": "Is X chaotic?", "ground_truth": "TRUE", "system_id": "lorenz63"},
        {"id": "core_2", "question": "Is Y periodic?", "ground_truth": "FALSE", "system_id": "shm"},
    ]
    with open(os.path.join(data_dir, "batch1_atomic_implication.jsonl"), "w") as f:
        for item in items_core:
            f.write(json.dumps(item) + "\n")

    # Adversarial (hard split)
    items_hard = [
        {"id": "adv_1", "question": "Tricky Q?", "ground_truth": "TRUE", "system_id": "lorenz63"},
    ]
    with open(os.path.join(data_dir, "batch10_adversarial.jsonl"), "w") as f:
        for item in items_hard:
            f.write(json.dumps(item) + "\n")

    # Heldout systems
    items_heldout = [
        {"id": "dysts_1", "question": "Is Z chaotic?", "ground_truth": "TRUE", "system_id": "dysts_halvorsen"},
        {"id": "dysts_2", "question": "Is W periodic?", "ground_truth": "FALSE", "system_id": "dysts_rossler"},
    ]
    with open(os.path.join(data_dir, "batch15_atomic_dysts.jsonl"), "w") as f:
        for item in items_heldout:
            f.write(json.dumps(item) + "\n")

    # Robustness
    items_robust = [
        {"id": "para_1", "question": "Paraphrase Q?", "ground_truth": "TRUE", "system_id": "lorenz63"},
    ]
    with open(os.path.join(data_dir, "batch11_consistency_paraphrase.jsonl"), "w") as f:
        for item in items_robust:
            f.write(json.dumps(item) + "\n")

    return data_dir


def test_assign_splits_no_overlap(tmp_path):
    """Items should not appear in multiple splits."""
    data_dir = _make_test_data_dir(tmp_path)
    splits = assign_splits(data_dir=data_dir)

    all_ids = []
    for split_name, items in splits.items():
        for item in items:
            all_ids.append(item.get("id"))

    assert len(all_ids) == len(set(all_ids)), "Duplicate item IDs across splits"


def test_validate_splits_clean(tmp_path):
    """validate_splits returns no errors for clean data."""
    data_dir = _make_test_data_dir(tmp_path)
    splits = assign_splits(data_dir=data_dir)
    errors = validate_splits(splits)
    assert errors == [], f"Unexpected validation errors: {errors}"


def test_validate_splits_detects_overlap():
    """validate_splits detects item ID overlap."""
    splits = {
        "core": [{"id": "item_1"}, {"id": "item_2"}],
        "hard": [{"id": "item_1"}],  # overlap!
        "robustness": [],
        "heldout_systems": [],
        "heldout_templates": [],
    }
    errors = validate_splits(splits)
    assert any("overlap" in e.lower() for e in errors)


def test_validate_splits_detects_leakage():
    """validate_splits detects dysts systems in core split."""
    splits = {
        "core": [{"id": "x", "system_id": "dysts_lorenz"}],  # leakage!
        "hard": [],
        "robustness": [],
        "heldout_systems": [],
        "heldout_templates": [],
    }
    errors = validate_splits(splits)
    assert any("leakage" in e.lower() for e in errors)


def test_compute_split_stats(tmp_path):
    """compute_split_stats returns per-split counts and hashes."""
    data_dir = _make_test_data_dir(tmp_path)
    splits = assign_splits(data_dir=data_dir)
    stats = compute_split_stats(splits)

    assert stats["core"]["item_count"] == 2
    assert stats["hard"]["item_count"] == 1
    assert stats["heldout_systems"]["item_count"] == 2
    assert stats["robustness"]["item_count"] == 1
    assert stats["total_items"] == 6
    assert "content_hash" in stats["core"]


def test_deterministic_assignment(tmp_path):
    """Same data + same seed = same split assignment."""
    data_dir = _make_test_data_dir(tmp_path)
    stats1 = compute_split_stats(assign_splits(data_dir=data_dir, seed=42))
    stats2 = compute_split_stats(assign_splits(data_dir=data_dir, seed=42))
    assert stats1 == stats2


def test_get_split_items_valid(tmp_path):
    """get_split_items returns items for a valid split."""
    data_dir = _make_test_data_dir(tmp_path)
    splits = assign_splits(data_dir=data_dir)
    core_items = get_split_items(splits, "core")
    assert len(core_items) == 2


def test_get_split_items_invalid():
    """get_split_items raises ValueError for invalid split."""
    with pytest.raises(ValueError, match="Invalid split"):
        get_split_items({}, "nonexistent_split")
