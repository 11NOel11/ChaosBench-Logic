"""Split assignment logic for ChaosBench-Logic v2.

Implements a 5-way split protocol for organizing benchmark items
into evaluation subsets with controlled system overlap.

Splits
------
- core: Batches 1-9, 12, 14 (original tasks) on core_30 systems
- robustness: Batches 10-11, 17 (perturbation variants)
- heldout_systems: Batches 15, 16, 18 (dysts-only, new systems)
- heldout_templates: Batch 11 template variants + new templates
- hard: Batch 10 adversarial + FOL edge cases + cross-indicator
"""

import hashlib
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


# Batch-to-split mapping
SPLIT_ASSIGNMENTS: Dict[str, str] = {
    # Core split: original task families on core systems
    "batch1_atomic_implication": "core",
    "batch2_multiHop_crossSystem": "core",
    "batch3_pde_chem_bio": "core",
    "batch4_maps_advanced": "core",
    "batch5_counterfactual_high_difficulty": "core",
    "batch6_deep_bias_probes": "core",
    "batch7_multiturn_advanced": "core",
    "batch8_indicator_diagnostics": "core",
    "batch9_regime_transitions": "core",
    "batch12_fol_inference": "core",
    "batch14_cross_indicator": "hard",
    # Robustness split: perturbation variants
    "batch10_adversarial": "hard",
    "batch11_consistency_paraphrase": "robustness",
    "batch13_extended_systems": "core",
    "batch17_perturbation_robustness": "robustness",
    # Heldout systems split: dysts-only batches
    "batch15_atomic_dysts": "heldout_systems",
    "batch16_multi_hop_dysts": "heldout_systems",
    "batch18_fol_dysts": "heldout_systems",
}

VALID_SPLITS = {"core", "robustness", "heldout_systems", "heldout_templates", "hard"}

# Core 30 system IDs (no dysts_ prefix)
CORE_SYSTEM_PREFIX_EXCLUDE = "dysts_"


def get_split_for_batch(batch_name: str) -> str:
    """Return the split name for a batch.

    Args:
        batch_name: Batch name without .jsonl extension.

    Returns:
        Split name string.

    Raises:
        ValueError: If batch is not in the split assignment table.
    """
    if batch_name in SPLIT_ASSIGNMENTS:
        return SPLIT_ASSIGNMENTS[batch_name]
    raise ValueError(f"Unknown batch: {batch_name}")


def assign_splits(
    data_dir: str = "data",
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Assign all items to splits based on batch membership.

    Args:
        data_dir: Directory containing batch JSONL files.
        seed: Random seed (used for deterministic heldout_templates assignment).

    Returns:
        Dict mapping split name to list of item dicts.
    """
    splits: Dict[str, List[Dict[str, Any]]] = {s: [] for s in VALID_SPLITS}

    pattern = re.compile(r"^batch(\d+)_.*\.jsonl$")
    batch_files = []
    if os.path.isdir(data_dir):
        for name in os.listdir(data_dir):
            m = pattern.match(name)
            if m:
                batch_files.append((int(m.group(1)), name))

    for _, filename in sorted(batch_files):
        batch_name = filename.replace(".jsonl", "")
        filepath = os.path.join(data_dir, filename)

        try:
            split = get_split_for_batch(batch_name)
        except ValueError:
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item["_split"] = split
                item["_batch"] = batch_name
                splits[split].append(item)

    return splits


def _collect_system_ids(items: List[Dict[str, Any]]) -> Set[str]:
    """Extract unique system IDs from a list of items."""
    return {
        item.get("system_id", "")
        for item in items
        if item.get("system_id")
    }


def _collect_item_ids(items: List[Dict[str, Any]]) -> Set[str]:
    """Extract unique item IDs from a list of items."""
    return {
        item.get("id", "")
        for item in items
        if item.get("id")
    }


def validate_splits(
    splits: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    """Validate split assignments for correctness.

    Checks:
    1. No item ID overlap between splits
    2. No heldout system leakage (dysts systems in core)
    3. Full coverage (every item assigned to exactly one split)
    4. Size constraints met

    Args:
        splits: Dict mapping split name to list of item dicts.

    Returns:
        List of validation error strings (empty if all checks pass).
    """
    errors: List[str] = []

    # Check 1: No item ID overlap
    all_ids_by_split: Dict[str, Set[str]] = {}
    for split_name, items in splits.items():
        ids = _collect_item_ids(items)
        all_ids_by_split[split_name] = ids

    split_names = list(all_ids_by_split.keys())
    for i, name_a in enumerate(split_names):
        for name_b in split_names[i + 1:]:
            overlap = all_ids_by_split[name_a] & all_ids_by_split[name_b]
            if overlap:
                sample = list(overlap)[:3]
                errors.append(
                    f"ID overlap between {name_a} and {name_b}: "
                    f"{len(overlap)} items (e.g., {sample})"
                )

    # Check 2: No heldout system leakage
    core_systems = _collect_system_ids(splits.get("core", []))
    heldout_systems = _collect_system_ids(splits.get("heldout_systems", []))

    dysts_in_core = {s for s in core_systems if s.startswith(CORE_SYSTEM_PREFIX_EXCLUDE)}
    if dysts_in_core:
        errors.append(
            f"Heldout system leakage: {len(dysts_in_core)} dysts systems in core split "
            f"(e.g., {list(dysts_in_core)[:3]})"
        )

    core_in_heldout = heldout_systems - {
        s for s in heldout_systems if s.startswith(CORE_SYSTEM_PREFIX_EXCLUDE)
    }
    # Allow "generic" system_id in heldout (for system-independent questions)
    core_in_heldout -= {"generic", ""}
    if core_in_heldout:
        errors.append(
            f"Core systems in heldout_systems split: {sorted(core_in_heldout)[:3]}"
        )

    # Check 3: Size constraints
    total_items = sum(len(items) for items in splits.values())
    if total_items == 0:
        errors.append("No items assigned to any split")

    for split_name in VALID_SPLITS:
        count = len(splits.get(split_name, []))
        if split_name in ("core", "heldout_systems") and count == 0 and total_items > 0:
            errors.append(f"Split '{split_name}' is empty")

    return errors


def compute_split_stats(
    splits: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Compute statistics for each split.

    Args:
        splits: Dict mapping split name to list of item dicts.

    Returns:
        Dict with per-split counts, system counts, and hashes.
    """
    stats: Dict[str, Any] = {}

    for split_name in VALID_SPLITS:
        items = splits.get(split_name, [])
        systems = _collect_system_ids(items)

        # Compute content hash for reproducibility
        content = json.dumps(
            [item.get("id", "") for item in items],
            sort_keys=True,
        )
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

        stats[split_name] = {
            "item_count": len(items),
            "system_count": len(systems),
            "systems": sorted(systems),
            "content_hash": content_hash,
        }

    stats["total_items"] = sum(
        stats[s]["item_count"] for s in VALID_SPLITS
    )

    return stats


def get_split_items(
    splits: Dict[str, List[Dict[str, Any]]],
    split_name: str,
) -> List[Dict[str, Any]]:
    """Get items for a specific split.

    Args:
        splits: Dict mapping split name to list of item dicts.
        split_name: Name of the split to retrieve.

    Returns:
        List of item dicts for the requested split.

    Raises:
        ValueError: If split_name is not valid.
    """
    if split_name not in VALID_SPLITS:
        raise ValueError(
            f"Invalid split: {split_name}. Valid splits: {sorted(VALID_SPLITS)}"
        )
    return splits.get(split_name, [])
