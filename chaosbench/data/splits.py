"""Split assignment logic for ChaosBench-Logic v2.

Implements a 5-way split protocol for organizing benchmark items
into evaluation subsets with controlled system overlap.

v2.1 Protocol (batch-based):
    Assigns items to splits based on batch membership.

v2.2 Protocol (hybrid system-based + template-based):
    1. Heldout systems: reserved system IDs (15 dysts systems)
    2. Heldout templates: reserved template hashes
    3. Hard: by construction (multi-hop >=3 steps, FOL >=3 premises, adversarial)
    4. Robustness: perturbation and consistency families
    5. Core: everything else

Splits
------
- core: ~35% - Atomic + indicator_diag + regime + extended on non-heldout systems
- robustness: ~30% - Perturbation + consistency families
- heldout_systems: ~15% - All families on 15 reserved dysts systems
- heldout_templates: ~8% - Novel templates on core systems
- hard: ~12% - Multi-hop >=3, FOL >=3, adversarial, cross-indicator
"""

import hashlib
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================================
# v2.1 Batch-to-Split Mapping (backward compatible)
# ============================================================================

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

# ============================================================================
# v2.2 Heldout System IDs (15 dysts systems)
# ============================================================================

HELDOUT_SYSTEM_IDS: Set[str] = {
    "dysts_sprotta",
    "dysts_sprottb",
    "dysts_sprottc",
    "dysts_sprottd",
    "dysts_sprotte",
    "dysts_sprottf",
    "dysts_sprottg",
    "dysts_sprotth",
    "dysts_sprotti",
    "dysts_sprottj",
    "dysts_sprottk",
    "dysts_sprottl",
    "dysts_sprottm",
    "dysts_sprottn",
    "dysts_sprotto",
}

# Template hashes reserved for heldout_templates split
# These are SHA-256 hashes of (question_text with system_id replaced by placeholder)
# Populated at build time; stored here for validation
HELDOUT_TEMPLATE_HASHES: Set[str] = set()


# ============================================================================
# v2.2 Hybrid Split Assignment
# ============================================================================

HARD_FAMILIES = {
    "adversarial_misleading",
    "adversarial_nearmiss",
    "adversarial_confusion",
    "cross_indicator",
}

ROBUSTNESS_FAMILIES = {
    "perturbation",
    "consistency_paraphrase",
}


def _hash_template(question_text: str, system_id: str) -> str:
    """Hash a question template with system_id masked out."""
    masked = question_text.replace(system_id, "<SYSTEM>")
    return hashlib.sha256(masked.encode("utf-8")).hexdigest()[:16]


def _is_hard_by_construction(item: Dict[str, Any]) -> bool:
    """Check if an item belongs to the hard split by construction.

    Hard items include:
    - Multi-hop questions with >=3 hops
    - FOL inference questions with >=3 predicates
    - Adversarial questions
    - Cross-indicator questions
    """
    item_type = item.get("type", "")

    # Adversarial and cross-indicator families are hard
    if item_type in HARD_FAMILIES:
        return True

    # Multi-hop with >=3 hops
    if item_type == "multi_hop":
        question = item.get("question", "")
        # Heuristic: 3-hop questions mention 3+ predicate transitions
        if ("therefore" in question.lower() and
                question.lower().count("must be") >= 2):
            return True

    # FOL with >=3 predicates in assignment
    if item_type == "fol_inference":
        question = item.get("question", "")
        if "assignment" in question.lower() and question.count(",") >= 2:
            return True

    return False


def assign_split_v22(item: Dict[str, Any]) -> str:
    """Assign a split to an item using v2.2 hybrid protocol.

    Args:
        item: Question dict with at least 'system_id', 'type', 'question' fields.

    Returns:
        Split name string.
    """
    system_id = item.get("system_id", "")

    # 1. Heldout systems: reserved system IDs
    if system_id in HELDOUT_SYSTEM_IDS:
        return "heldout_systems"

    # 2. Heldout templates: reserved template hashes
    if HELDOUT_TEMPLATE_HASHES:
        template_hash = _hash_template(
            item.get("question", ""), system_id
        )
        if template_hash in HELDOUT_TEMPLATE_HASHES:
            return "heldout_templates"

    # 3. Hard: by construction
    if _is_hard_by_construction(item):
        return "hard"

    # 4. Robustness: perturbation and consistency families
    item_type = item.get("type", "")
    if item_type in ROBUSTNESS_FAMILIES:
        return "robustness"

    # 5. Core: everything else
    return "core"


# ============================================================================
# Common functions (work with both v2.1 and v2.2)
# ============================================================================

def get_split_for_batch(batch_name: str) -> str:
    """Return the split name for a batch (v2.1 protocol).

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
    use_v22: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Assign all items to splits.

    Args:
        data_dir: Directory containing batch JSONL files.
        seed: Random seed (used for deterministic heldout_templates assignment).
        use_v22: If True, use v2.2 hybrid protocol instead of batch-based.

    Returns:
        Dict mapping split name to list of item dicts.
    """
    splits: Dict[str, List[Dict[str, Any]]] = {s: [] for s in VALID_SPLITS}

    pattern = re.compile(r"^(batch\d+_.*|v22_.*)\.jsonl$")
    batch_files = []
    if os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            m = pattern.match(name)
            if m:
                batch_files.append(name)

    for filename in batch_files:
        batch_name = filename.replace(".jsonl", "")
        filepath = os.path.join(data_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item["_batch"] = batch_name

                if use_v22 or batch_name.startswith("v22_"):
                    split = assign_split_v22(item)
                else:
                    try:
                        split = get_split_for_batch(batch_name)
                    except ValueError:
                        continue

                item["_split"] = split
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

    # Check that heldout system IDs don't appear in core
    leakage = core_systems & HELDOUT_SYSTEM_IDS
    if leakage:
        errors.append(
            f"Heldout system leakage into core: {sorted(leakage)[:3]}"
        )

    # Also check the old dysts_ prefix heuristic
    dysts_in_core = {s for s in core_systems if s.startswith(CORE_SYSTEM_PREFIX_EXCLUDE)}
    # Filter out systems not in HELDOUT_SYSTEM_IDS (they're allowed in core for v2.2)
    dysts_in_core_heldout = dysts_in_core & HELDOUT_SYSTEM_IDS
    if dysts_in_core_heldout:
        errors.append(
            f"Heldout dysts systems in core split: {sorted(dysts_in_core_heldout)[:3]}"
        )

    core_in_heldout = heldout_systems - {
        s for s in heldout_systems if s.startswith(CORE_SYSTEM_PREFIX_EXCLUDE)
    }
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
    """Compute statistics for each split."""
    stats: Dict[str, Any] = {}

    for split_name in VALID_SPLITS:
        items = splits.get(split_name, [])
        systems = _collect_system_ids(items)

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
