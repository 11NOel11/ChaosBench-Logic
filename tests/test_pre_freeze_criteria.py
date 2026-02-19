"""Tests for pre-freeze quality criteria (HARD FAIL enforcement).

These tests enforce the quality standard and must pass before freezing an API subset.
"""

import json
import os
from collections import defaultdict

import pytest

from chaosbench.data.grouping import _normalize_text
from chaosbench.data.splits import assign_split_v22


@pytest.fixture
def v22_questions():
    """Load all v2.2 questions."""
    questions = []
    data_dir = "data"
    for fname in sorted(os.listdir(data_dir)):
        if fname.startswith('v22_') and fname.endswith('.jsonl'):
            with open(os.path.join(data_dir, fname)) as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))
    return questions


def test_zero_accidental_duplicates(v22_questions):
    """HARD FAIL: Accidental duplicates must be zero."""
    seen_keys = {}
    accidental_dupes = []

    for q in v22_questions:
        norm_text = _normalize_text(q['question'])
        key = f"{q['type']}:{q['system_id']}:{norm_text}:{q['ground_truth']}"

        if key in seen_keys:
            # Check if this is intentional (has group_id)
            has_group = 'group_id' in q or 'group_id' in seen_keys[key]
            if not has_group:
                accidental_dupes.append((seen_keys[key]['id'], q['id']))
        else:
            seen_keys[key] = q

    assert len(accidental_dupes) == 0, \
        f"Found {len(accidental_dupes)} accidental duplicates: {accidental_dupes[:5]}"


def test_overall_balance_40_60(v22_questions):
    """HARD FAIL: Overall label balance must be 40-60% TRUE."""
    true_count = sum(1 for q in v22_questions if q['ground_truth'] == 'TRUE')
    total = len(v22_questions)
    true_pct = true_count / total * 100 if total > 0 else 0

    assert 40 <= true_pct <= 60, \
        f"Overall balance {true_pct:.1f}% outside 40-60% range"


def test_zero_label_leakage(v22_questions):
    """HARD FAIL: Label leakage must be zero."""
    import re

    forbidden_patterns = [
        r'\bground_truth\b',
        r'\banswer_is\b',
        r'\bcorrect answer\b',
        r'\bthe answer is (TRUE|FALSE)\b',
        r'\(TRUE\)',
        r'\(FALSE\)',
    ]

    leaks = []
    for q in v22_questions:
        text = q['question'].lower()
        for pattern in forbidden_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                leaks.append((q['id'], pattern, q['question'][:100]))
                break

    assert len(leaks) == 0, \
        f"Found {len(leaks)} label leaks: {leaks[:3]}"


def test_split_no_item_id_collisions(v22_questions):
    """HARD FAIL: No item_id should appear in multiple splits."""
    all_ids = [q['id'] for q in v22_questions]
    unique_ids = set(all_ids)

    collisions = len(all_ids) - len(unique_ids)
    assert collisions == 0, \
        f"Found {collisions} item_id collisions across splits"


def test_split_no_system_leakage(v22_questions):
    """HARD FAIL: Heldout systems must not appear in other splits."""
    # Assign splits
    for q in v22_questions:
        q['_split'] = assign_split_v22(q)

    # Group by split
    splits = defaultdict(list)
    for q in v22_questions:
        splits[q['_split']].append(q)

    # Check system leakage
    if 'heldout_systems' in splits:
        heldout_systems = set(q['system_id'] for q in splits['heldout_systems'])
        other_systems = set(q['system_id'] for q in v22_questions if q['_split'] != 'heldout_systems')
        system_leakage = heldout_systems & other_systems

        assert len(system_leakage) == 0, \
            f"Found {len(system_leakage)} system IDs leaking from heldout_systems: {list(system_leakage)[:10]}"


def test_split_no_text_leakage_core_heldout(v22_questions):
    """HARD FAIL: No normalized text should leak between core and heldout_templates."""
    # Assign splits
    for q in v22_questions:
        q['_split'] = assign_split_v22(q)

    # Group by split
    splits = defaultdict(list)
    for q in v22_questions:
        splits[q['_split']].append(q)

    # Check text leakage between core and heldout_templates
    if 'core' in splits and 'heldout_templates' in splits:
        core_texts = set(_normalize_text(q['question']) for q in splits['core'])
        heldout_texts = set(_normalize_text(q['question']) for q in splits['heldout_templates'])
        text_leakage = core_texts & heldout_texts

        assert len(text_leakage) == 0, \
            f"Found {len(text_leakage)} normalized texts leaking between core and heldout_templates"


def test_all_splits_non_degenerate(v22_questions):
    """HARD FAIL: All splits must have ≥10% minority class."""
    # Assign splits
    for q in v22_questions:
        q['_split'] = assign_split_v22(q)

    # Group by split
    splits = defaultdict(list)
    for q in v22_questions:
        splits[q['_split']].append(q)

    # Check minority class per split
    degenerate_splits = []
    for split_name, split_qs in splits.items():
        if len(split_qs) == 0:
            continue

        true_count = sum(1 for q in split_qs if q['ground_truth'] == 'TRUE')
        total = len(split_qs)
        true_pct = true_count / total * 100 if total > 0 else 0
        minority_pct = min(true_pct, 100 - true_pct)

        if minority_pct < 10:
            degenerate_splits.append((split_name, minority_pct, total))

    assert len(degenerate_splits) == 0, \
        f"Found {len(degenerate_splits)} degenerate splits (< 10% minority): {degenerate_splits}"


def test_multi_hop_not_degenerate_ci_smoke():
    """HARD FAIL (CI smoke): Multi-hop must have ≥10% minority class."""
    # Load multi-hop questions
    multi_hop_path = "data/v22_multi_hop.jsonl"
    if not os.path.exists(multi_hop_path):
        pytest.skip("Multi-hop file not found")

    questions = []
    with open(multi_hop_path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    true_count = sum(1 for q in questions if q['ground_truth'] == 'TRUE')
    total = len(questions)
    true_pct = true_count / total * 100 if total > 0 else 0
    minority_pct = min(true_pct, 100 - true_pct)

    assert minority_pct >= 10, \
        f"Multi-hop is degenerate: {minority_pct:.1f}% minority class (need ≥10%)"


def test_manifest_has_required_fields():
    """SOFT: Manifest should have core required fields."""
    manifest_path = "data/v2_manifest.json"
    if not os.path.exists(manifest_path):
        pytest.skip("Manifest not found")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Core required fields (top-level)
    required_fields = ['schema_version', 'timestamp', 'generation_config', 'batches']

    # Seed can be either top-level or in generation_config
    has_seed = 'seed' in manifest or ('generation_config' in manifest and 'seed' in manifest['generation_config'])

    missing = [f for f in required_fields if f not in manifest]

    assert len(missing) == 0, \
        f"Manifest missing required fields: {missing}"

    assert has_seed, \
        "Manifest missing seed (check both top-level and generation_config)"


def test_determinism_via_question_count():
    """SOFT: Question counts should be stable across builds."""
    # This is a weak determinism check - just verify counts match expected
    manifest_path = "data/v2_manifest.json"
    if not os.path.exists(manifest_path):
        pytest.skip("Manifest not found")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Check that batch counts in manifest match actual files
    for batch_name, batch_info in manifest.get('batches', {}).items():
        if not batch_name.startswith('v22_'):
            continue

        expected_count = batch_info['count']
        filepath = f"data/{batch_name}.jsonl"

        if not os.path.exists(filepath):
            continue

        actual_count = sum(1 for line in open(filepath) if line.strip())

        assert actual_count == expected_count, \
            f"Batch {batch_name}: manifest says {expected_count}, file has {actual_count}"
