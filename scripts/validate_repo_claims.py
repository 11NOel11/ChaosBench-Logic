#!/usr/bin/env python3
"""
Validate that repository documentation claims match actual dataset statistics.

This script is used in CI to ensure README.md, DATASET.md, and other docs
stay in sync with the actual dataset.

Usage:
    python scripts/validate_repo_claims.py
    
Exit codes:
    0 = All claims validated
    1 = Validation failed (mismatch found)
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict


def compute_stats():
    """Compute actual dataset statistics."""
    
    batch_files = [
        "data/batch1_atomic_implication.jsonl",
        "data/batch2_multiHop_crossSystem.jsonl",
        "data/batch3_pde_chem_bio.jsonl",
        "data/batch4_maps_advanced.jsonl",
        "data/batch5_counterfactual_high_difficulty.jsonl",
        "data/batch6_deep_bias_probes.jsonl",
        "data/batch7_multiturn_advanced.jsonl",
    ]
    
    all_items = []
    for batch_file in batch_files:
        with open(batch_file) as f:
            for line in f:
                if line.strip():
                    all_items.append(json.loads(line))
    
    # Basic counts
    total_items = len(all_items)
    unique_ids = len(set(item['id'] for item in all_items))
    
    # Task types
    task_types = Counter(item.get('type') for item in all_items)
    num_task_types = len(task_types)
    
    # Systems
    systems = Counter(item.get('system_id') for item in all_items if 'system_id' in item)
    systems_used = len(systems)
    
    systems_dir = Path("systems")
    all_system_files = list(systems_dir.glob("*.json"))
    systems_defined = len(all_system_files)
    
    # Dialogues
    dialogues = defaultdict(list)
    for item in all_items:
        if 'dialogue_id' in item:
            dialogues[item['dialogue_id']].append(item)
    
    num_dialogues = len(dialogues)
    
    # Predicates (from systems files)
    if all_system_files:
        with open(all_system_files[0]) as f:
            first_system = json.load(f)
            if 'truth_assignment' in first_system:
                num_predicates = len(first_system['truth_assignment'])
            else:
                num_predicates = None
    else:
        num_predicates = None
    
    return {
        'total_items': total_items,
        'unique_ids': unique_ids,
        'num_task_types': num_task_types,
        'systems_used': systems_used,
        'systems_defined': systems_defined,
        'num_dialogues': num_dialogues,
        'num_predicates': num_predicates,
    }


def validate_published_results():
    """Validate published_results structure and prevent raw outputs from being tracked."""

    published_dir = Path("published_results")
    errors = []

    if not published_dir.exists():
        errors.append("published_results/ directory not found")
        return errors

    # Canonical configurations (must exist for official publication)
    canonical_configs = [
        "claude3_zeroshot",
        "gemini_zeroshot",
        "gpt4_cot",
        "gpt4_zeroshot",
        "llama3_cot",
        "llama3_zeroshot",
    ]

    # Required files per configuration
    required_files = [
        "summary.json",
        "run_meta.json",
        "accuracy_by_task.csv",
        "metrics_overview.csv",
    ]

    # Forbidden raw/debug artifacts (should never appear in published_results/)
    forbidden_names = {
        "per_item_results.jsonl",
        "debug_samples.jsonl",
        "checkpoint.json",
        "run.log",
        "bias_errors.csv",
    }

    forbidden_globs = [
        "*.jsonl",
        "*.log",
        "*.png",
        "*.jpg",
        "*.jpeg",
    ]

    # Validate every config directory that exists under published_results/
    config_dirs = [d for d in published_dir.iterdir() if d.is_dir()]
    if not config_dirs:
        errors.append("No configuration directories found in published_results/")
        return errors

    for config_dir in config_dirs:
        config_name = config_dir.name

        # Required files
        for filename in required_files:
            if not (config_dir / filename).exists():
                errors.append(f"Missing published_results/{config_name}/{filename}")

        # Forbidden exact names
        for filename in forbidden_names:
            if (config_dir / filename).exists():
                errors.append(
                    f"Forbidden raw output file found: published_results/{config_name}/{filename}"
                )

        # Forbidden patterns
        for pattern in forbidden_globs:
            for match in config_dir.glob(pattern):
                errors.append(
                    f"Forbidden file found: published_results/{config_name}/{match.name}"
                )

    # Ensure canonical configs exist (for official published results)
    for cfg in canonical_configs:
        if not (published_dir / cfg).is_dir():
            errors.append(f"Missing canonical configuration: published_results/{cfg}/")

    return errors


def validate_claims():
    """Validate repo claims against actual statistics."""

    stats = compute_stats()

    print("Validating ChaosBench-Logic repository claims...")
    print()

    errors = []
    
    # Check 1: Total items
    if stats['total_items'] != 621:
        errors.append(f"Total items mismatch: expected 621, got {stats['total_items']}")
    else:
        print(f"✓ Total items: {stats['total_items']}")
    
    # Check 2: Unique IDs
    if stats['unique_ids'] != 621:
        errors.append(f"Unique IDs mismatch: expected 621, got {stats['unique_ids']}")
    else:
        print(f"✓ Unique IDs: {stats['unique_ids']}")
    
    # Check 3: Task types
    if stats['num_task_types'] != 17:
        errors.append(f"Task types mismatch: expected 17, got {stats['num_task_types']}")
    else:
        print(f"✓ Task types: {stats['num_task_types']}")
    
    # Check 4: Systems used
    if stats['systems_used'] != 27:
        errors.append(f"Systems used mismatch: expected 27, got {stats['systems_used']}")
    else:
        print(f"✓ Systems used in dataset: {stats['systems_used']}")
    
    # Check 5: Systems defined
    if stats['systems_defined'] != 30:
        errors.append(f"Systems defined mismatch: expected 30, got {stats['systems_defined']}")
    else:
        print(f"✓ Systems defined: {stats['systems_defined']}")
    
    # Check 6: Dialogues
    if stats['num_dialogues'] != 49:
        errors.append(f"Dialogues mismatch: expected 49, got {stats['num_dialogues']}")
    else:
        print(f"✓ Multi-turn dialogues: {stats['num_dialogues']}")
    
    # Check 7: Predicates
    if stats['num_predicates'] and stats['num_predicates'] != 11:
        errors.append(f"Predicates mismatch: expected 11, got {stats['num_predicates']}")
    elif stats['num_predicates']:
        print(f"✓ Predicates per system: {stats['num_predicates']}")

    # Check 8: Published results structure
    print()
    print("Validating published_results/ structure...")
    results_errors = validate_published_results()
    if results_errors:
        errors.extend(results_errors)
    else:
        print("✓ Published results structure validated")

    print()

    if errors:
        print("VALIDATION FAILED:")
        for error in errors:
            print(f"  ✗ {error}")
        return False
    else:
        print("All claims validated successfully!")
        return True


if __name__ == "__main__":
    success = validate_claims()
    sys.exit(0 if success else 1)
