"""Tests for the 5-way split protocol."""

import json
import os
import tempfile

import pytest

from chaosbench.data.splits import (
    HELDOUT_SYSTEM_IDS,
    SPLIT_ASSIGNMENTS,
    VALID_SPLITS,
    assign_split_v22,
    assign_splits,
    compute_split_stats,
    get_split_for_batch,
    get_split_items,
    validate_splits,
    _hash_template,
    _is_hard_by_construction,
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
    """validate_splits detects heldout dysts systems in core split."""
    splits = {
        "core": [{"id": "x", "system_id": "dysts_sprotta"}],  # leakage! (heldout system in core)
        "hard": [],
        "robustness": [],
        "heldout_systems": [],
        "heldout_templates": [],
    }
    errors = validate_splits(splits)
    assert any("leakage" in e.lower() or "heldout" in e.lower() for e in errors)


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


# ============================================================================
# v2.2 Split Protocol Tests
# ============================================================================


class TestAssignSplitV22:
    """Tests for the v2.2 hybrid split assignment."""

    def test_heldout_system_assigned_correctly(self):
        """Items from heldout systems go to heldout_systems split."""
        item = {"system_id": "dysts_sprotta", "type": "atomic", "question": "Is it chaotic?"}
        assert assign_split_v22(item) == "heldout_systems"

    def test_all_heldout_systems_recognized(self):
        """All 15 heldout system IDs route to heldout_systems."""
        for sys_id in HELDOUT_SYSTEM_IDS:
            item = {"system_id": sys_id, "type": "atomic", "question": "Q?"}
            assert assign_split_v22(item) == "heldout_systems"

    def test_adversarial_goes_to_hard(self):
        """Adversarial items go to hard split."""
        for adv_type in ("adversarial_misleading", "adversarial_nearmiss", "adversarial_confusion"):
            item = {"system_id": "lorenz63", "type": adv_type, "question": "Tricky?"}
            assert assign_split_v22(item) == "hard"

    def test_cross_indicator_goes_to_hard(self):
        """Cross-indicator items go to hard split."""
        item = {"system_id": "lorenz63", "type": "cross_indicator", "question": "Q?"}
        assert assign_split_v22(item) == "hard"

    def test_perturbation_goes_to_robustness(self):
        """Perturbation items go to robustness split."""
        item = {"system_id": "lorenz63", "type": "perturbation", "question": "Q?"}
        assert assign_split_v22(item) == "robustness"

    def test_consistency_goes_to_robustness(self):
        """Consistency paraphrase items go to robustness split."""
        item = {"system_id": "lorenz63", "type": "consistency_paraphrase", "question": "Q?"}
        assert assign_split_v22(item) == "robustness"

    def test_atomic_goes_to_core(self):
        """Atomic items on non-heldout systems go to core."""
        item = {"system_id": "lorenz63", "type": "atomic", "question": "Is Lorenz chaotic?"}
        assert assign_split_v22(item) == "core"

    def test_indicator_diagnostics_goes_to_core(self):
        """Indicator diagnostics items go to core."""
        item = {"system_id": "lorenz63", "type": "indicator_diagnostics", "question": "Q?"}
        assert assign_split_v22(item) == "core"

    def test_heldout_system_overrides_hard_type(self):
        """Heldout system assignment takes priority over hard family type."""
        item = {"system_id": "dysts_sprottb", "type": "adversarial_misleading", "question": "Q?"}
        assert assign_split_v22(item) == "heldout_systems"

    def test_heldout_system_overrides_robustness_type(self):
        """Heldout system assignment takes priority over robustness family type."""
        item = {"system_id": "dysts_sprottc", "type": "perturbation", "question": "Q?"}
        assert assign_split_v22(item) == "heldout_systems"

    def test_dysts_non_heldout_goes_to_core(self):
        """Dysts systems NOT in heldout set go to core for atomic items."""
        item = {"system_id": "dysts_aizawa", "type": "atomic", "question": "Q?"}
        assert assign_split_v22(item) == "core"


class TestIsHardByConstruction:
    """Tests for the hard-by-construction heuristic."""

    def test_adversarial_types_are_hard(self):
        for t in ("adversarial_misleading", "adversarial_nearmiss", "adversarial_confusion"):
            assert _is_hard_by_construction({"type": t})

    def test_cross_indicator_is_hard(self):
        assert _is_hard_by_construction({"type": "cross_indicator"})

    def test_atomic_is_not_hard(self):
        assert not _is_hard_by_construction({"type": "atomic", "question": "Simple?"})

    def test_multi_hop_3_hops_is_hard(self):
        q = "X must be Y, therefore Z must be W, therefore this must be true"
        assert _is_hard_by_construction({"type": "multi_hop", "question": q})

    def test_fol_3_predicates_is_hard(self):
        q = "Given assignment: Chaotic=T, Deterministic=T, PosLyap=T, is X?"
        assert _is_hard_by_construction({"type": "fol_inference", "question": q})


class TestHashTemplate:
    """Tests for the template hashing function."""

    def test_same_template_different_systems(self):
        """Same template with different system IDs produces same hash."""
        h1 = _hash_template("Is lorenz63 chaotic?", "lorenz63")
        h2 = _hash_template("Is rossler chaotic?", "rossler")
        assert h1 == h2

    def test_different_templates_different_hash(self):
        """Different templates produce different hashes."""
        h1 = _hash_template("Is lorenz63 chaotic?", "lorenz63")
        h2 = _hash_template("Is lorenz63 periodic?", "lorenz63")
        assert h1 != h2

    def test_hash_is_deterministic(self):
        h1 = _hash_template("Is lorenz63 chaotic?", "lorenz63")
        h2 = _hash_template("Is lorenz63 chaotic?", "lorenz63")
        assert h1 == h2


class TestAssignSplitsV22Mode:
    """Tests for assign_splits with use_v22=True and v22_ file prefix."""

    def test_v22_prefix_files_use_v22_protocol(self, tmp_path):
        """Files prefixed with v22_ use the v2.2 split protocol."""
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir)

        items = [
            {"id": "a1", "question": "Q?", "ground_truth": "TRUE", "type": "atomic", "system_id": "lorenz63"},
            {"id": "a2", "question": "Q?", "ground_truth": "FALSE", "type": "perturbation", "system_id": "lorenz63"},
            {"id": "a3", "question": "Q?", "ground_truth": "TRUE", "type": "atomic", "system_id": "dysts_sprotta"},
        ]
        with open(os.path.join(data_dir, "v22_atomic.jsonl"), "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        splits = assign_splits(data_dir=data_dir)
        assert len(splits["core"]) == 1
        assert splits["core"][0]["id"] == "a1"
        assert len(splits["robustness"]) == 1
        assert splits["robustness"][0]["id"] == "a2"
        assert len(splits["heldout_systems"]) == 1
        assert splits["heldout_systems"][0]["id"] == "a3"

    def test_use_v22_flag_overrides_batch_protocol(self, tmp_path):
        """use_v22=True forces v2.2 protocol even for batch-named files."""
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir)

        # Write an adversarial item in a batch file; normally batch10 -> "hard"
        # With v2.2, an atomic item on a non-heldout system -> "core"
        items = [
            {"id": "x1", "question": "Q?", "ground_truth": "TRUE", "type": "atomic", "system_id": "lorenz63"},
        ]
        with open(os.path.join(data_dir, "batch10_adversarial.jsonl"), "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        splits = assign_splits(data_dir=data_dir, use_v22=True)
        # v2.2 protocol: atomic on lorenz63 -> core (not "hard" from batch assignment)
        assert len(splits["core"]) == 1
